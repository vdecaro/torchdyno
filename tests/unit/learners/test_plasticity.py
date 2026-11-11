import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tests.unit.compositions import (
    adadiag,
    reservoir,
)
from torchdyno.layers import (
    Recurrent,
    Stack,
)
from torchdyno.learners import (
    IntrinsicPlasticity,
    Ridge,
)

DTYPE = torch.float64


def _expected_update(cell, x, lr, mu, sigma):
    """The intrinsic plasticity rule over one batch, written from its equations."""
    w_in, b_in, w = cell.input.weight, cell.input.bias, cell.recurrent.weight
    gain, bias = cell.gain.clone(), cell.bias.clone()
    state = torch.zeros(x.shape[1], cell.state_size, dtype=x.dtype)
    var = sigma**2
    for u in x:
        pre = u @ w_in.T + b_in + state @ w.T
        out = torch.tanh(gain * pre + bias)
        state = (1 - cell.leak) * state + cell.leak * out
        d_bias = -mu / var + (out / var) * (2 * var + 1 - out**2 + mu * out)
        d_gain = -1 / gain + d_bias * pre
        gain, bias = gain - lr * d_gain.mean(0), bias - lr * d_bias.mean(0)
    return gain, bias


def test_one_batch_follows_the_update_rule():
    torch.manual_seed(0)
    model = nn.Sequential(reservoir(2, 16, dtype=DTYPE), nn.Linear(16, 1, dtype=DTYPE))
    cell = model[0].cell
    x = torch.randn(25, 3, 2, dtype=DTYPE)
    gain, bias = _expected_update(cell, x, lr=0.1, mu=0.1, sigma=0.3)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    IntrinsicPlasticity(lr=0.1, mu=0.1, sigma=0.3).fit(model, [(x, None)])
    torch.testing.assert_close(cell.gain, gain)
    torch.testing.assert_close(cell.bias, bias)
    changed = {
        k for k, v in model.state_dict().items() if not torch.equal(before[k], v)
    }
    assert changed == {"0.cell.gain", "0.cell.bias"}
    assert not cell._forward_hooks


def test_adapts_every_cell_of_a_deep_reservoir_even_when_frozen():
    torch.manual_seed(0)
    deep = Stack(reservoir(1, 8), reservoir(8, 8)).requires_grad_(False)
    model = nn.Sequential(deep, nn.Linear(16, 1))
    before = [layer.cell.gain.clone() for layer in deep]
    IntrinsicPlasticity(lr=0.05, epochs=2).fit(model, [(torch.randn(20, 4, 1), None)])
    assert all(
        not torch.equal(layer.cell.gain, g)
        for layer, g in zip(deep, before, strict=True)
    )


def test_then_ridge_fits_the_readout():
    torch.manual_seed(0)
    model = nn.Sequential(reservoir(1, 32), nn.Linear(32, 1))
    x = torch.randn(200, 2, 1)
    IntrinsicPlasticity(lr=0.01, epochs=3).fit(model, [(x, None)])
    Ridge(l2=1e-4, washout=20).fit(model, [(x, torch.roll(x, 1, 0))])
    error = (model(x)[20:] - torch.roll(x, 1, 0)[20:]).pow(2).mean()
    assert error < 0.1 * x.var()


def test_needs_tanh_leaky_cells():
    with pytest.raises(ValueError, match="gain, a bias"):
        IntrinsicPlasticity().fit(nn.Sequential(adadiag(), nn.Linear(12, 1)), [])
    model = nn.Sequential(reservoir(), nn.Linear(16, 1))
    model[0].cell.activation = torch.relu
    with pytest.raises(ValueError, match="tanh activations"):
        IntrinsicPlasticity().fit(model, [])


@pytest.mark.parametrize("activation", [F.tanh, nn.Tanh()])
def test_accepts_every_form_of_tanh(activation):
    model = nn.Sequential(reservoir(), nn.Linear(16, 1))
    model[0].cell.activation = activation
    IntrinsicPlasticity().fit(model, [(torch.randn(5, 2, 2), None)])


def test_needs_a_positive_target_spread():
    with pytest.raises(ValueError, match="sigma"):
        IntrinsicPlasticity(sigma=0.0)


class ScaledCell(nn.Module):
    """A cell of the user's own with the gain, bias and preactivation IP adapts."""

    def __init__(self):
        super().__init__()
        self.state_size = 4
        self.input = nn.Linear(2, 4)
        self.gain = nn.Parameter(torch.ones(4))
        self.bias = nn.Parameter(torch.zeros(4))
        self.activation = nn.Tanh()

    def preactivation(self, u, h):
        return self.input(u) + 0.5 * h

    def forward(self, u, h):
        return self.activation(self.gain * self.preactivation(u, h) + self.bias)


def test_adapts_any_cell_with_a_gain_a_bias_and_a_preactivation():
    model = nn.Sequential(Recurrent(ScaledCell()), nn.Linear(4, 1))
    IntrinsicPlasticity(lr=0.05).fit(model, [(torch.randn(10, 3, 2), None)])
    assert not torch.equal(model[0].cell.gain, torch.ones(4))
