import pytest
import torch
from torch import nn

from tests.unit.compositions import reservoir
from torchdyno.learners import (
    RLS,
    Ridge,
)

DTYPE = torch.float64


def test_one_pass_equals_ridge_without_bias():
    torch.manual_seed(0)
    model = nn.Sequential(
        reservoir(1, 16, dtype=DTYPE), nn.Linear(16, 1, bias=False, dtype=DTYPE)
    )
    x = torch.randn(60, 2, 1, dtype=DTYPE)
    y = torch.roll(x, 3, dims=0)
    RLS(l2=0.1, washout=5).fit(model, [(x[:, :1], y[:, :1]), (x[:, 1:], y[:, 1:])])
    online = model[-1].weight.detach().clone()
    Ridge(l2=0.1, washout=5).fit(model, [(x, y)])
    torch.testing.assert_close(online, model[-1].weight, rtol=1e-8, atol=1e-10)


def test_bias_is_the_weight_of_a_constant_feature():
    torch.manual_seed(0)
    x = torch.randn(40, 5, dtype=DTYPE)
    y = x @ torch.randn(5, 2, dtype=DTYPE) + 1.5
    model = nn.Sequential(nn.Identity(), nn.Linear(5, 2, dtype=DTYPE))
    RLS(l2=0.5).fit(model, [(x, y)])
    rows = torch.cat([x, torch.ones(40, 1, dtype=DTYPE)], 1)
    expected = torch.linalg.solve(
        rows.T @ rows + 0.5 * torch.eye(6, dtype=DTYPE), rows.T @ y
    )
    torch.testing.assert_close(model[-1].weight, expected[:5].T)
    torch.testing.assert_close(model[-1].bias, expected[5])


def test_needs_a_linear_readout():
    with pytest.raises(TypeError, match=r"nn\.Linear"):
        RLS().fit(nn.Sequential(nn.Identity(), nn.Identity()), [])


def _stream(seed=0, n=60):
    torch.manual_seed(seed)
    x = torch.randn(n, 4, dtype=DTYPE)
    return x, x @ torch.randn(4, 1, dtype=DTYPE) + 0.1 * torch.randn(n, 1, dtype=DTYPE)


def _readout():
    return nn.Sequential(nn.Identity(), nn.Linear(4, 1, bias=False, dtype=DTYPE))


def test_later_fits_continue_the_stream_until_reset():
    x, y = _stream()
    split, whole, learner = _readout(), _readout(), RLS(l2=0.1)
    learner.fit(split, [(x[:30], y[:30])])
    learner.fit(split, [(x[30:], y[30:])])
    RLS(l2=0.1).fit(whole, [(x, y)])
    torch.testing.assert_close(split[-1].weight, whole[-1].weight)
    learner.reset()
    learner.fit(split, [(x[30:], y[30:])])
    RLS(l2=0.1).fit(whole, [(x[30:], y[30:])])
    torch.testing.assert_close(split[-1].weight, whole[-1].weight)


def test_forgetting_discounts_older_samples_geometrically():
    x, y = _stream()
    model, decay = _readout(), 0.95
    RLS(l2=0.1, forgetting=decay).fit(model, [(x, y)])
    weights = decay ** torch.arange(len(x) - 1, -1, -1, dtype=DTYPE)
    prior = 0.1 * decay ** len(x) * torch.eye(4, dtype=DTYPE)
    gram = x.T @ (weights[:, None] * x) + prior
    expected = torch.linalg.solve(gram, x.T @ (weights[:, None] * y))
    torch.testing.assert_close(model[-1].weight, expected.T)


def test_update_is_the_step_fit_takes():
    x, y = _stream()
    by_fit, by_update = _readout(), _readout()
    RLS(l2=0.1).fit(by_fit, [(x, y)])
    learner = RLS(l2=0.1)
    for t in range(len(x)):
        learner.update(by_update[-1], x[t : t + 1], y[t : t + 1])
    torch.testing.assert_close(by_fit[-1].weight, by_update[-1].weight)


def test_rejects_forgetting_out_of_range_and_a_readout_of_another_size():
    with pytest.raises(ValueError, match="forgetting"):
        RLS(forgetting=0.0)
    learner = RLS()
    learner.fit(_readout(), [_stream()])
    wider = nn.Sequential(nn.Identity(), nn.Linear(5, 1, bias=False, dtype=DTYPE))
    with pytest.raises(ValueError, match="reset"):
        learner.fit(
            wider, [(torch.randn(8, 5, dtype=DTYPE), torch.randn(8, 1, dtype=DTYPE))]
        )
