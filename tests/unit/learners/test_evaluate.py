import pytest
import torch
from torch import nn

from tests.unit.compositions import reservoir
from torchdyno.learners import (
    Backprop,
    Ridge,
    evaluate,
)
from torchdyno.pooling import Last


def _seen(parts, batches):
    torch.manual_seed(0)
    seen = {}

    def probe(pred, target):
        seen["pred"], seen["target"] = pred, target
        return 0.0

    evaluate(nn.Sequential(*parts), batches, probe)
    return seen["pred"], seen["target"]


def test_scores_time_aligned_outputs_as_one_row_per_timestep():
    # sequences of different lengths combine row by row
    batches = [
        (torch.randn(t, b, 1), torch.randn(t, b, 2)) for t, b in ((5, 3), (4, 2))
    ]
    pred, target = _seen([reservoir(1, 8), nn.Linear(8, 2)], batches)
    assert pred.shape == (5 * 3 + 4 * 2, 2)
    assert torch.equal(target, torch.cat([y.flatten(0, 1) for _, y in batches]))


def test_concatenates_pooled_outputs_on_dim_zero():
    batches = [(torch.randn(5, b, 1), torch.randn(b, 2)) for b in (3, 2)]
    pred, target = _seen([reservoir(1, 8), Last(), nn.Linear(8, 2)], batches)
    assert pred.shape == (5, 2)
    assert torch.equal(target, torch.cat([y for _, y in batches]))


def _dropout_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Dropout(0.5), Last(), nn.Linear(4, 1)).train()


def test_ridge_fits_on_features_without_dropout():
    x, y = torch.randn(3, 64, 4), torch.randn(64, 1)
    model, reference = _dropout_model(), _dropout_model().eval()
    Ridge(1e-3).fit(model, [(x, y)])
    Ridge(1e-3).fit(reference, [(x, y)])
    torch.testing.assert_close(model[-1].weight, reference[-1].weight)


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize(
    "fit",
    [
        lambda m, d: Ridge(1e-3).fit(m, d),
        lambda m, d: evaluate(m, d, lambda p, t: 0.0),
        lambda m, d: Backprop(nn.MSELoss(), epochs=1).fit(m, d),
        lambda m, d: Backprop(nn.MSELoss(), epochs=1, score_fn=lambda p, t: 0.0).fit(
            m, d, d
        ),
    ],
)
def test_leaves_every_module_in_the_mode_it_found(fit, training):
    model = _dropout_model().train(training)
    model[0].train(not training)
    fit(model, [(torch.randn(3, 8, 4), torch.randn(8, 1))])
    assert [m.training for m in model] == [not training, training, training]


def test_leaves_the_washout_unscored():
    seen = {}

    def probe(pred, target):
        seen["pred"], seen["target"] = pred, target
        return 0.0

    x, y = torch.randn(6, 2, 3), torch.randn(6, 2, 3)
    evaluate(nn.Identity(), [(x, y)], probe, washout=4)
    assert torch.equal(seen["pred"], x[4:].flatten(0, 1))
    assert torch.equal(seen["target"], y[4:].flatten(0, 1))


@pytest.mark.parametrize(
    ("shape", "washout"), [((6, 2, 3), 6), ((6, 2, 3), -1), ((2, 3), 1)]
)
def test_rejects_a_washout_it_cannot_apply(shape, washout):
    with pytest.raises(ValueError, match="washout"):
        evaluate(
            nn.Identity(),
            [(torch.randn(shape), torch.randn(shape))],
            torch.equal,
            washout,
        )
