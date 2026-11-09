import pytest
import torch
from torch import nn

from tests.unit.compositions import reservoir
from torchdyno.learners import Ridge
from torchdyno.pooling import Last

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


def test_needs_a_linear_readout_and_validation_to_select():
    with pytest.raises(TypeError, match=r"nn\.Linear"):
        Ridge().fit(nn.Sequential(nn.Identity(), nn.Identity()), [])
    with pytest.raises(ValueError, match="needs val"):
        Ridge(l2=[1e-6, 1e-3]).fit(_model(2, 1, bias=True), [])
