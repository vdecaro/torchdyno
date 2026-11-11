"""Each composition against a loop written from its update equation."""

import math

import pytest
import torch
from torch import nn

from tests.unit.compositions import (
    adadiag,
    cfc,
    cornn,
    reservoir,
    scn,
)
from torchdyno.cells import (
    ClosedFormCell,
    EulerCell,
    Gate,
    LeakyCell,
    LeakyField,
    LinearDiagonalCell,
    OscillatorCell,
    RingDiagonal,
)
from torchdyno.layers import (
    Recurrent,
    Scan,
    Stack,
)
from torchdyno.matrices import Dense
from torchdyno.testing import assert_layer_conforms

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
    field, gate = cell.field, cell.step
    beta, w_in = field.recurrent.weight, field.input.weight
    coupling = _skew(field.coupling)

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
    field = cell.field
    blocks = torch.tanh(field.recurrent.mask * field.recurrent.weight)
    coupling, w_in = _skew(field.coupling), field.input.weight

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
    field = LeakyField(
        input=Dense(2, 6, dtype=DTYPE), recurrent=Dense(6, 6, dtype=DTYPE)
    )
    cell = EulerCell(field, step=0.5)
    w_in, w = field.input.weight, field.recurrent.weight

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


def test_cfc():
    torch.manual_seed(0)
    layer = cfc(features=2, size=5, units=7, dtype=DTYPE)
    cell = layer.cell
    w_back, b_back = cell.backbone[0].weight, cell.backbone[0].bias
    w_heads, b_heads = cell.heads.weight, cell.heads.bias

    def step(u, h):
        f = torch.tanh(torch.cat([h, u], -1) @ w_back.T + b_back)
        f1, f2, t_a, t_b = (
            f @ w_heads[i * 5 : (i + 1) * 5].T + b_heads[i * 5 : (i + 1) * 5]
            for i in range(4)
        )
        gate = torch.sigmoid(t_a * 1.0 + t_b)
        return torch.tanh(f1) * (1 - gate) + torch.tanh(f2) * gate

    _assert_matches(layer, step, torch.randn(20, 3, 2, dtype=DTYPE))


def test_closed_form_cell_needs_a_positive_step():
    with pytest.raises(ValueError, match="dt must be > 0"):
        ClosedFormCell(nn.Identity(), nn.Linear(6, 16), dt=0.0)


def _field():
    return LeakyField(Dense(2, 4), Dense(4, 4))


@pytest.mark.parametrize(
    ("make", "name"),
    [
        (lambda: LeakyCell(Dense(2, 4), Dense(4, 4), leak=torch.rand(4)), "leak"),
        (
            lambda: EulerCell(_field(), step=torch.rand(4)),
            "step",
        ),
    ],
)
def test_tensor_hyperparameters_follow_the_cell(make, name):
    cell, fresh = make(), make()
    fresh.load_state_dict(cell.state_dict())
    assert torch.equal(getattr(fresh, name), getattr(cell, name))
    assert getattr(cell.double(), name).dtype == torch.float64


MISMATCHED = {
    "field input": lambda: LeakyField(Dense(2, 3), Dense(4, 4)),
    "field recurrent": lambda: LeakyField(Dense(2, 4), Dense(3, 4)),
    "field coupling": lambda: LeakyField(Dense(2, 4), Dense(4, 4), Dense(4, 3)),
    "euler gate": lambda: EulerCell(_field(), step=Gate(2, 1)),
    "euler step tensor": lambda: EulerCell(_field(), step=torch.ones(3)),
    "leaky input": lambda: LeakyCell(Dense(2, 1), Dense(4, 4)),
    "leaky leak tensor": lambda: LeakyCell(
        Dense(2, 4), Dense(4, 4), leak=torch.ones(2)
    ),
    "oscillator recurrent": lambda: OscillatorCell(Dense(2, 4), Dense(4, 4)),
    "oscillator input": lambda: OscillatorCell(Dense(2, 3), Dense(8, 4)),
    "closed form heads": lambda: ClosedFormCell(nn.Identity(), Dense(6, 6)),
    "linear diagonal": lambda: LinearDiagonalCell(
        RingDiagonal(3), Dense(2, 4, dtype=torch.cfloat)
    ),
}


@pytest.mark.parametrize("make", MISMATCHED.values(), ids=MISMATCHED.keys())
def test_cells_reject_parts_that_do_not_match_the_state(make):
    with pytest.raises(ValueError, match=r"width|four equal parts"):
        make()


OUT_OF_RANGE = {
    "gate bounds": lambda: Gate(2, 4, bounds=(0.2, 0.1)),
    "euler step": lambda: EulerCell(_field(), step=0.0),
    "euler step tensor": lambda: EulerCell(
        _field(), step=torch.tensor([0.1, 0.1, -0.1, 0.1])
    ),
    "leak above one": lambda: LeakyCell(Dense(2, 4), Dense(4, 4), leak=1.5),
    "leak of zero": lambda: LeakyCell(Dense(2, 4), Dense(4, 4), leak=0.0),
    "oscillator nan dt": lambda: OscillatorCell(Dense(2, 4), Dense(8, 4), dt=math.nan),
    "closed form nan dt": lambda: ClosedFormCell(
        nn.Identity(), Dense(6, 8), dt=math.nan
    ),
}


@pytest.mark.parametrize("make", OUT_OF_RANGE.values(), ids=OUT_OF_RANGE.keys())
def test_cells_reject_constants_out_of_range(make):
    with pytest.raises(ValueError, match=r"must be|Need"):
        make()


def test_leaky_cell_takes_a_matrix_without_parameters():
    class Identity(nn.Module):
        in_features = out_features = 4

        def forward(self, x):
            return x

    cell = LeakyCell(Dense(2, 4), Identity())
    assert cell.gain.dtype == torch.get_default_dtype()


def test_scan_needs_a_linear_cell():
    with pytest.raises(TypeError, match="Scan needs a linear cell"):
        Scan(EulerCell(_field()))


def test_euler_integrates_any_field_with_any_step():
    # an antisymmetric RNN: h' = h + eps tanh((W - W^T - gamma I) h + V u)
    class Antisymmetric(nn.Module):
        state_size = 4

        def __init__(self):
            super().__init__()
            self.w, self.v = nn.Parameter(torch.randn(4, 4)), Dense(2, 4)

        def forward(self, u, h):
            w = self.w - self.w.T - 0.1 * torch.eye(4)
            return torch.tanh(h @ w.T + self.v(u))

    layer = Recurrent(EulerCell(Antisymmetric(), step=lambda u, h: 0.1 + 0 * h))
    assert_layer_conforms(layer, torch.randn(6, 3, 2))


def test_euler_needs_the_width_of_a_field_without_one():
    with pytest.raises(ValueError, match="state_size"):
        EulerCell(lambda u, h: -h)
    assert EulerCell(lambda u, h: -h, state_size=3).state_size == 3


def test_oscillator_and_closed_form_steps_follow_the_same_contract():
    # per-unit tensors, and an elapsed time read from the input for irregular sampling
    oscillator = OscillatorCell(Dense(2, 4), Dense(8, 4), gamma=torch.rand(4) + 0.5)
    cfc = ClosedFormCell(nn.Identity(), Dense(6, 16), dt=lambda u, h: u[..., :1].exp())
    for cell in (oscillator, cfc):
        assert_layer_conforms(Recurrent(cell), torch.randn(6, 3, 2))
