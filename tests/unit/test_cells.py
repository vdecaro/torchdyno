"""Each composition against a loop written from its update equation."""

import pytest
import torch
from torch import nn

from tests.unit.compositions import (
    adadiag,
    cornn,
    reservoir,
    scn,
)
from torchdyno.cells import (
    EulerCell,
    OscillatorCell,
)
from torchdyno.layers import (
    Recurrent,
    Stack,
)
from torchdyno.matrices import Dense

DTYPE = torch.float64


def _loop(step, x, size):
    h = torch.zeros(x.shape[1], size, dtype=x.dtype)
    states = []
    for u in x:
        h = step(u, h)
        states.append(h)
    return torch.stack(states)


def _assert_matches(layer, step, x):
    expected = _loop(step, x, layer.cell.state_size)
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=1e-10)


def _skew(coupling):
    c = coupling.mask * coupling.weight
    return c - c.T


def test_adadiag():
    torch.manual_seed(0)
    layer = adadiag(dtype=DTYPE)
    cell = layer.cell
    with torch.no_grad():
        cell.step.state.normal_()  # starts at zero; exercise the state term
    beta, w_in, gate = cell.recurrent.weight, cell.input.weight, cell.step
    coupling = _skew(cell.coupling)

    def step(u, v):
        logits = u @ gate.input.weight.T + gate.input.bias + gate.state * v
        nu = 1e-5 + (0.2 - 1e-5) * torch.sigmoid(logits)
        drive = -v + torch.tanh(beta) * torch.relu(v) + v @ coupling.T + u @ w_in.T
        return v + nu * drive

    _assert_matches(layer, step, torch.randn(20, 3, 2, dtype=DTYPE))


def test_scn():
    torch.manual_seed(0)
    layer = scn(step=0.1, dtype=DTYPE)
    cell = layer.cell
    blocks = torch.tanh(cell.recurrent.mask * cell.recurrent.weight)
    coupling, w_in = _skew(cell.coupling), cell.input.weight

    def step(u, v):
        drive = -v + torch.relu(v) @ blocks.T + v @ coupling.T + u @ w_in.T
        return v + 0.1 * drive

    _assert_matches(layer, step, torch.randn(20, 3, 2, dtype=DTYPE))


def _reservoir_step(cell):
    w_in, b, w = cell.input.weight, cell.input.bias, cell.recurrent.weight

    def step(u, h):
        pre = u @ w_in.T + b + h @ w.T
        return (1 - 0.5) * h + 0.5 * torch.tanh(cell.gain * pre + cell.bias)

    return step


def test_reservoir():
    torch.manual_seed(0)
    layer = reservoir(leak=0.5, dtype=DTYPE)
    with torch.no_grad():  # start at 1 and 0; exercise them
        layer.cell.gain.uniform_(0.5, 1.5)
        layer.cell.bias.normal_(0.0, 0.1)
    _assert_matches(
        layer, _reservoir_step(layer.cell), torch.randn(20, 3, 2, dtype=DTYPE)
    )


def test_deep_reservoir_reads_every_layer_or_the_last():
    torch.manual_seed(0)
    first, second = reservoir(2, 16, dtype=DTYPE), reservoir(16, 8, dtype=DTYPE)
    x = torch.randn(20, 3, 2, dtype=DTYPE)
    h1 = _loop(_reservoir_step(first.cell), x, 16)
    h2 = _loop(_reservoir_step(second.cell), h1, 8)
    every = Stack(first, second)(x)
    torch.testing.assert_close(every, torch.cat([h1, h2], -1), rtol=0, atol=1e-10)
    torch.testing.assert_close(nn.Sequential(first, second)(x), every[..., 16:])


def test_gain_and_bias_start_as_a_no_op():
    torch.manual_seed(0)
    layer = reservoir(dtype=DTYPE)
    cell = layer.cell
    x = torch.randn(10, 2, 2, dtype=DTYPE)

    def plain(u, h):
        return 0.5 * h + 0.5 * torch.tanh(cell.preactivation(u, h))

    torch.testing.assert_close(layer(x), _loop(plain, x, 16), rtol=0, atol=0)


def test_euler_cell_without_coupling():
    torch.manual_seed(0)
    cell = EulerCell(
        input=Dense(2, 6, dtype=DTYPE), recurrent=Dense(6, 6, dtype=DTYPE), step=0.5
    )
    w_in, w = cell.input.weight, cell.recurrent.weight

    def step(u, v):
        return v + 0.5 * (-v + torch.relu(v) @ w.T + u @ w_in.T)

    _assert_matches(Recurrent(cell), step, torch.randn(10, 3, 2, dtype=DTYPE))


def test_cornn():
    torch.manual_seed(0)
    layer = cornn(features=2, size=6, dtype=DTYPE)
    cell = layer.cell
    w, w_z = cell.recurrent.weight[:, :6], cell.recurrent.weight[:, 6:]
    v, b = cell.input.weight, cell.input.bias
    with torch.no_grad():
        b.normal_()  # starts at zero; exercise it

    def step(u, state):
        y, z = state[:, :6], state[:, 6:]
        drive = torch.tanh(y @ w.T + z @ w_z.T + u @ v.T + b)
        z = z + 0.1 * (drive - 1.0 * y - 1.0 * z)
        return torch.cat([y + 0.1 * z, z], -1)

    _assert_matches(layer, step, torch.randn(20, 3, 2, dtype=DTYPE))


def test_oscillator_needs_positive_constants():
    with pytest.raises(ValueError, match="must be > 0"):
        OscillatorCell(Dense(2, 4), Dense(8, 4), dt=0.0)
