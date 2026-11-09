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
