import pytest
import torch
from torch import nn

from tests.unit.compositions import reservoir
from torchdyno.layers import Concat
from torchdyno.learners import (
    RLS,
    Ridge,
)
from torchdyno.matrices import (
    BlockDiagonal,
    Dense,
    Diagonal,
)
from torchdyno.metrics import Accuracy
from torchdyno.pooling import (
    Last,
    Mean,
)

DTYPE = torch.float64


def _model(features, outputs, bias):
    return nn.Sequential(
        nn.Identity(), nn.Linear(features, outputs, bias=bias, dtype=DTYPE)
    )


def _reference(x, y, l2, intercept=False, standardize=False):
    """argmin ||Y - X W - 1 b||² + l2 ||W||², b unpenalized, X optionally scaled."""
    mean_x = x.mean(0) if intercept else torch.zeros(x.shape[1], dtype=x.dtype)
    mean_y = y.mean(0) if intercept else torch.zeros(y.shape[1], dtype=y.dtype)
    scale = (
        (x - x.mean(0)).pow(2).mean(0).sqrt() if standardize else torch.ones(x.shape[1])
    )
    xs = (x - mean_x) / scale
    coef = torch.linalg.solve(
        xs.T @ xs + l2 * torch.eye(x.shape[1], dtype=x.dtype), xs.T @ (y - mean_y)
    )
    weight = coef / scale[:, None]
    return weight.T, mean_y - mean_x @ weight


def _assert_fits(model, x, y, l2, **reference):
    weight, bias = _reference(x, y, l2, **reference)
    torch.testing.assert_close(model[-1].weight, weight)
    if model[-1].bias is not None:
        torch.testing.assert_close(model[-1].bias, bias)


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("intercept", [False, True])
@pytest.mark.parametrize(
    "n", [40, 6]
)  # more samples than features (primal), fewer (dual)
def test_matches_the_closed_form(n, intercept, standardize):
    torch.manual_seed(0)
    x = (
        torch.randn(n, 10, dtype=DTYPE) * torch.linspace(0.5, 3.0, 10, dtype=DTYPE)
        + 1.0
    )
    y = x @ torch.randn(10, 2, dtype=DTYPE) + 0.5
    model = _model(10, 2, bias=intercept)
    Ridge(l2=1e-2, standardize=standardize).fit(
        model, [(x[: n // 2], y[: n // 2]), (x[n // 2 :], y[n // 2 :])]
    )
    _assert_fits(model, x, y, 1e-2, intercept=intercept, standardize=standardize)


def test_batches_crossing_from_dual_to_primal():
    torch.manual_seed(0)
    x, y = torch.randn(30, 10, dtype=DTYPE), torch.randn(30, 1, dtype=DTYPE)
    model = _model(10, 1, bias=True)
    Ridge(l2=1e-1).fit(model, [(x[i : i + 3], y[i : i + 3]) for i in range(0, 30, 3)])
    _assert_fits(model, x, y, 1e-1, intercept=True)


def test_time_aligned_features_with_washout():
    torch.manual_seed(0)
    x, y = torch.randn(20, 3, 4, dtype=DTYPE), torch.randn(20, 3, 2, dtype=DTYPE)
    model = _model(4, 2, bias=False)
    Ridge(l2=1e-3, washout=5).fit(model, [(x, y)])
    _assert_fits(model, x[5:].reshape(-1, 4), y[5:].reshape(-1, 2), 1e-3)


def test_washout_needs_time_aligned_features():
    model = _model(4, 2, bias=False)
    with pytest.raises(ValueError, match="time-aligned"):
        Ridge(washout=2).fit(
            model, [(torch.randn(3, 4, dtype=DTYPE), torch.randn(3, 2, dtype=DTYPE))]
        )


def test_classifies_pooled_states_against_one_hot_labels():
    torch.manual_seed(0)
    model = nn.Sequential(
        reservoir(1, 32, dtype=DTYPE), Last(), nn.Linear(32, 2, dtype=DTYPE)
    )
    before = {k: v.clone() for k, v in model[0].state_dict().items()}
    x = torch.randn(10, 64, 1, dtype=DTYPE)
    labels = (x[-1, :, 0] > 0).long()
    Ridge(l2=1e-3).fit(model, [(x, labels)])
    features = model[:-1](x).detach()
    _assert_fits(
        model, features, nn.functional.one_hot(labels, 2).double(), 1e-3, intercept=True
    )
    assert (model(x).argmax(-1) == labels).double().mean() > 0.9
    assert all(torch.equal(model[0].state_dict()[k], v) for k, v in before.items())


@pytest.mark.parametrize("learner", [Ridge(), RLS()])
@pytest.mark.parametrize(
    ("features", "labels"),
    [
        ((8, 4), torch.zeros(8)),  # float class indices
        ((5, 8, 4), torch.zeros(8, dtype=torch.long)),  # per-sequence labels, unpooled
        ((8, 4), torch.zeros(8, 1, dtype=torch.long)),  # an extra label dimension
    ],
)
def test_rejects_targets_that_do_not_match_the_readout(learner, features, labels):
    model = nn.Sequential(nn.Identity(), nn.Linear(4, 3))
    with pytest.raises(ValueError, match="don't match"):
        learner.fit(model, [(torch.randn(*features), labels)])


@pytest.mark.parametrize("learner", [Ridge(), RLS()])
def test_rejects_features_that_are_not_finite(learner):
    x = torch.randn(10, 8, 4)
    x[:3, 0] = torch.inf  # a state that diverged
    model = nn.Sequential(Mean(), nn.Linear(4, 3))
    with pytest.raises(ValueError, match="1 feature rows are not finite"):
        learner.fit(model, [(x, torch.zeros(8, dtype=torch.long))])


@pytest.mark.parametrize("learner", [Ridge(1e-3), RLS(1e-3)])
@pytest.mark.parametrize("n", [40, 3])  # primal and dual
def test_solves_on_the_device_of_the_features(learner, n):
    x, y = torch.randn(n, 4, dtype=DTYPE), torch.randn(n, 2, dtype=DTYPE)
    model = _model(4, 2, bias=True)
    with torch.device("meta"):  # where tensors made without a device would land
        learner.fit(model, [(x, y)])
    assert model[-1].weight.isfinite().all()


@pytest.mark.parametrize("learner", [Ridge(1e-2, readout="1"), RLS(1e-2, readout="1")])
def test_fits_a_readout_in_the_middle_on_its_own_input(learner):
    torch.manual_seed(0)
    x = torch.randn(40, 4, dtype=DTYPE)
    y = x @ torch.randn(4, 2, dtype=DTYPE)
    model = nn.Sequential(
        nn.Identity(), nn.Linear(4, 2, bias=False, dtype=DTYPE), nn.Tanh()
    )
    learner.fit(model, [(x, y)])
    _assert_fits(model[:2], x, y, 1e-2)


def test_fits_a_dense_readout():
    torch.manual_seed(0)
    x = torch.randn(40, 4, dtype=DTYPE)
    y = x @ torch.randn(4, 2, dtype=DTYPE) + 0.5
    model = nn.Sequential(nn.Identity(), Dense(4, 2, bias=nn.init.zeros_, dtype=DTYPE))
    Ridge(l2=1e-2).fit(model, [(x, y)])
    weight, bias = _reference(x, y, 1e-2, intercept=True)
    torch.testing.assert_close(model[-1].weight, weight)
    torch.testing.assert_close(model[-1].bias, bias)


@pytest.mark.parametrize(
    "model",
    [
        nn.Sequential(nn.Identity(), Diagonal(4)),
        nn.Sequential(nn.Identity(), BlockDiagonal([2, 2])),
        nn.Sequential(nn.Identity(), Dense(4, 4, dtype=torch.cfloat)),
        nn.Sequential(nn.Linear(4, 4), nn.Tanh()),
    ],
)
def test_rejects_readouts_that_are_not_real_linear_maps(model):
    with pytest.raises(TypeError, match=r"real nn\.Linear or Dense"):
        Ridge().fit(model, [(torch.randn(8, 4), torch.randn(8, 4))])


def test_needs_a_name_for_the_readout_of_a_non_sequential_model():
    with pytest.raises(TypeError, match="readout="):
        Ridge().fit(nn.Linear(4, 2), [(torch.randn(8, 4), torch.randn(8, 2))])


def test_rejects_a_readout_that_runs_more_than_once():
    shared = nn.Linear(4, 2)
    model = nn.Sequential(nn.Identity(), Concat(shared, shared))
    with pytest.raises(ValueError, match="ran 2"):
        Ridge(readout="1.0").fit(model, [(torch.randn(8, 4), torch.randn(8, 2))])


def test_selects_l2_on_validation():
    torch.manual_seed(0)
    x = torch.randn(40, 6, dtype=DTYPE)
    y = x @ torch.randn(6, 1, dtype=DTYPE) + 0.1 * torch.randn(40, 1, dtype=DTYPE)

    def mse(pred, target):
        return torch.mean((pred - target) ** 2).item()

    model = _model(6, 1, bias=True)
    result = Ridge(l2=[1e-6, 1e2, 1e4], score_fn=mse, mode="min").fit(
        model, [(x[:30], y[:30])], [(x[30:], y[30:])]
    )
    assert result.best["l2"] == 1e-6
    assert result.best["score"] == pytest.approx(mse(model(x[30:]), y[30:]))
    assert result.history["l2"] == [1e-6, 1e2, 1e4]
    assert min(result.history["val_score"]) == result.best["score"]


def test_scores_validation_after_the_washout():
    torch.manual_seed(0)
    x = torch.randn(30, 2, 6, dtype=DTYPE)
    y = x @ torch.randn(6, 1, dtype=DTYPE)
    y[:5] += 100.0  # a transient the readout never fits

    def mse(pred, target):
        return torch.mean((pred - target) ** 2).item()

    model = _model(6, 1, bias=True)
    result = Ridge(l2=[1e-6, 1e-3], washout=5, score_fn=mse, mode="min").fit(
        model, [(x, y)], [(x, y)]
    )
    assert result.best["score"] == pytest.approx(mse(model(x)[5:], y[5:]))


@pytest.mark.parametrize("l2", [(1e-6, 1e2), torch.tensor([1e-6, 1e2])])
def test_takes_candidates_as_any_sequence_or_tensor(l2):
    torch.manual_seed(0)
    x = torch.randn(40, 2, dtype=DTYPE)
    y = x @ torch.ones(2, 1, dtype=DTYPE)
    result = Ridge(
        l2, score_fn=lambda p, t: (p - t).abs().max().item(), mode="min"
    ).fit(_model(2, 1, bias=False), [(x, y)], [(x, y)])
    assert result.best["l2"] == pytest.approx(1e-6)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"l2": []}, "at least one"),
        ({"l2": -1.0}, ">= 0"),
        ({"mode": "maximize"}, "mode"),
    ],
)
def test_rejects_penalties_and_modes_it_cannot_use(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Ridge(**kwargs)


def test_selects_in_the_direction_of_the_metric():
    torch.manual_seed(0)
    x = torch.randn(60, 4, dtype=DTYPE)
    y = (x[:, 0] > 0).long()
    model = _model(4, 2, bias=True)
    result = Ridge([1e-3, 1e6], score_fn=Accuracy()).fit(model, [(x, y)], [(x, y)])
    assert result.best == {"l2": 1e-3, "score": Accuracy()(model(x), y)}


def test_needs_a_linear_readout_and_validation_to_select():
    with pytest.raises(TypeError, match=r"nn\.Linear"):
        Ridge().fit(nn.Sequential(nn.Identity(), nn.Identity()), [])
    with pytest.raises(ValueError, match="needs val"):
        Ridge(l2=[1e-6, 1e-3]).fit(_model(2, 1, bias=True), [])
