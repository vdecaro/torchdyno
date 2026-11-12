"""LRU, S4D and ParalESN against loops written from their recurrences."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tests.unit.compositions import (
    lru,
    paralesn,
    s4d,
)
from torchdyno.matrices import (
    Dense,
    Diagonal,
)
from torchdyno.sequence import (
    LeakyDiagonal,
    LinearDiagonalCell,
    Recurrent,
    RingDiagonal,
    Scan,
    ZOHDiagonal,
)
from torchdyno.testing import assert_layer_conforms

DTYPE = torch.float64


def _loop(a, s, w_in, x):
    h = torch.zeros(x.shape[1], a.shape[0], dtype=a.dtype)
    states = []
    for u in x:
        h = a * h + s * (u.to(w_in.dtype) @ w_in.T)
        states.append(torch.cat([h.real, h.imag], -1))
    return torch.stack(states)


def test_lru():
    torch.manual_seed(0)
    layer = lru(features=3, modes=6, dtype=DTYPE)
    ring, w_in = layer.cell.transition, layer.cell.input.dense()
    lam = torch.exp(-torch.exp(ring.nu)) * torch.exp(1j * ring.theta)
    gamma = torch.sqrt(1 - lam.abs() ** 2)
    x = torch.randn(25, 4, 3, dtype=DTYPE)
    torch.testing.assert_close(layer(x), _loop(lam, gamma, w_in, x), rtol=0, atol=1e-10)


def test_s4d():
    torch.manual_seed(0)
    layer = s4d(features=3, modes=6, dtype=DTYPE)
    zoh, w_in = layer.cell.transition, layer.cell.input.dense()
    a = -torch.exp(zoh.log_real) + 1j * zoh.imag
    a_bar = torch.exp(torch.exp(zoh.log_dt) * a)
    x = torch.randn(25, 4, 3, dtype=DTYPE)
    torch.testing.assert_close(
        layer(x), _loop(a_bar, (a_bar - 1) / a, w_in, x), rtol=0, atol=1e-10
    )


def test_paralesn():
    torch.manual_seed(0)
    layer = paralesn(features=3, modes=6, leak=0.3, dtype=DTYPE)
    lam, w_in = layer.cell.transition.diagonal.diagonal(), layer.cell.input.dense()
    x = torch.randn(25, 4, 3, dtype=DTYPE)
    expected = _loop(0.7 + 0.3 * lam, 0.3, w_in, x)
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=1e-10)
    torch.testing.assert_close(Scan(layer.cell)(x), expected)


def test_leaky_diagonal_checks_its_leak():
    diagonal = Diagonal(4, dtype=torch.cfloat)
    for leak in (0.0, 1.5, torch.tensor([0.5, 0.5, 0.5, 1.2])):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            LeakyDiagonal(diagonal, leak)
    with pytest.raises(ValueError, match="leak has width 3"):
        LeakyDiagonal(diagonal, torch.full((3,), 0.5))
    per_mode = LeakyDiagonal(diagonal, torch.full((4,), 0.5))
    assert "leak" in dict(per_mode.named_buffers())


def test_ring_draws_magnitudes_and_phases_within_bounds():
    torch.manual_seed(0)
    lam, gamma = RingDiagonal(2000, r_min=0.4, r_max=0.9, max_phase=math.pi / 4)()
    assert lam.abs().min() >= 0.4 - 1e-6 and lam.abs().max() <= 0.9 + 1e-6
    assert lam.angle().min() >= 0 and lam.angle().max() <= math.pi / 4 + 1e-6
    torch.testing.assert_close(gamma**2 + lam.abs() ** 2, torch.ones(2000))
    with pytest.raises(ValueError, match="r_min <= r_max"):
        RingDiagonal(4, r_min=0.9, r_max=0.4)


def test_zero_order_hold_starts_from_s4d_lin():
    torch.manual_seed(0)
    zoh = ZOHDiagonal(500, dt_min=1e-4, dt_max=1e-1)
    torch.testing.assert_close(-torch.exp(zoh.log_real), torch.full((500,), -0.5))
    torch.testing.assert_close(zoh.imag, math.pi * torch.arange(500.0))
    dt = torch.exp(zoh.log_dt)
    assert dt.min() >= 1e-4 * (1 - 1e-6) and dt.max() <= 1e-1 * (1 + 1e-6)
    with pytest.raises(ValueError, match="dt_min <= dt_max"):
        ZOHDiagonal(4, dt_min=0.0)


def test_zero_order_hold_scale_is_accurate_for_tiny_steps():
    zoh = ZOHDiagonal(4, dt_min=1e-7, dt_max=1e-7)
    _, scale = zoh()
    assert scale.dtype == torch.complex64
    torch.testing.assert_close(scale.real, torch.full((4,), 1e-7), rtol=1e-4, atol=0)


@pytest.mark.parametrize("build", [lru, s4d])
def test_training_keeps_the_transition_stable(build):
    torch.manual_seed(0)
    layer = build()
    opt = torch.optim.Adam(layer.parameters(), lr=1.0)
    before = layer.cell.input.weight.detach().clone()
    for _ in range(5):
        opt.zero_grad()
        layer(torch.randn(10, 3, 2)).pow(2).sum().backward()
        opt.step()
    a, _ = layer.cell.transition()
    assert (a.abs() < 1).all()
    assert not torch.equal(layer.cell.input.weight, before)


def test_ring_rejects_a_ring_with_no_room_below_one():
    with pytest.raises(ValueError, match="r_min < 1"):
        RingDiagonal(4, r_min=1.0, r_max=1.0)
    assert RingDiagonal(10_000, r_min=0.999, r_max=1.0).nu.isfinite().all()


def test_zero_order_hold_keeps_its_scale_as_a_vanishes():
    zoh = ZOHDiagonal(3, dt_min=1e-2, dt_max=1e-2)
    with torch.no_grad():
        zoh.log_real.copy_(torch.tensor([-30.0, -45.0, -80.0]))
        zoh.imag.zero_()
    _, scale = zoh()
    torch.testing.assert_close(scale, torch.full((3,), 1e-2, dtype=scale.dtype))


class RealDiagonal(nn.Module):
    """A real, input-independent transition: a = sigmoid(w), s = 1 - a."""

    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(size, dtype=DTYPE))

    def forward(self, u=None):  # noqa: ARG002
        a = torch.sigmoid(self.weight)
        return a, 1 - a


class Selective(nn.Module):
    """An input-dependent step, as in selective state spaces."""

    def __init__(self, features, size):
        super().__init__()
        self.step = nn.Linear(features, size, dtype=DTYPE)

    def forward(self, u):
        a = torch.exp(-F.softplus(self.step(u)))
        return a, 1 - a


@pytest.mark.parametrize(
    "make",
    [
        lambda: RealDiagonal(5),
        lambda: Selective(2, 5),
        lambda: LeakyDiagonal(Diagonal(5, bound=torch.tanh, dtype=DTYPE), 0.3),
    ],
)
@pytest.mark.parametrize("layer", [Recurrent, Scan])
def test_real_and_input_dependent_transitions_run_in_both_layers(make, layer):
    torch.manual_seed(0)
    cell = LinearDiagonalCell(Dense(2, 5, dtype=DTYPE), make())
    assert cell.state_size == 5
    assert_layer_conforms(layer(cell), torch.randn(9, 3, 2, dtype=DTYPE))
