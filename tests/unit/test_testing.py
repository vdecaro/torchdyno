import pytest
import torch
from torch import nn

from tests.unit.compositions import (
    COMPOSITIONS,
    reservoir,
)
from torchdyno.matrices import Dense
from torchdyno.sequence import (
    Cell,
    Recurrent,
)
from torchdyno.testing import (
    assert_layer_conforms,
    echo_state_distance,
)


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", COMPOSITIONS)
def test_compositions_conform(name, dtype, frozen):
    torch.manual_seed(0)
    layer = COMPOSITIONS[name](dtype=dtype).requires_grad_(not frozen)
    assert_layer_conforms(layer, torch.randn(8, 3, 2, dtype=dtype))


class SumCell(Cell):
    """h' = h + W u."""

    def __init__(self):
        super().__init__()
        self.input = Dense(3, 3)
        self.state_size = 3

    def forward(self, u, h):
        return h + self.input(u)


class WrongWidth(Recurrent):
    def forward(self, x, h0=None):
        return super().forward(x, h0)[..., :2]


class IgnoresH0(Recurrent):
    def forward(self, x, h0=None):  # noqa: ARG002
        return super().forward(x)


class Detached(Recurrent):
    def forward(self, x, h0=None):
        return super().forward(x.detach(), h0)


class NotTheLoop(Recurrent):
    def forward(self, x, h0=None):
        return 2 * super().forward(x, h0)


class MixesTheBatch(Cell):
    """h' = h + W u + the batch mean of h: sequences leak into each other."""

    def __init__(self):
        super().__init__()
        self.input = Dense(3, 3)
        self.state_size = 3

    def forward(self, u, h):
        return h + self.input(u) + h.mean(0)


class DropsOut(SumCell):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)

    def forward(self, u, h):
        return self.dropout(super().forward(u, h))


class DeadParameter(Recurrent):
    def __init__(self, cell):
        super().__init__(cell)
        self.unused = nn.Parameter(torch.zeros(3))


def test_accepts_a_plain_recurrent_layer():
    x = torch.randn(6, 2, 3)
    assert_layer_conforms(Recurrent(SumCell()), x)


@pytest.mark.parametrize(
    "layer_cls", [WrongWidth, IgnoresH0, Detached, NotTheLoop, DeadParameter]
)
def test_rejects_a_broken_layer(layer_cls):
    with pytest.raises(AssertionError):
        assert_layer_conforms(layer_cls(SumCell()), torch.randn(6, 2, 3))


def test_rejects_a_cell_that_mixes_the_batch():
    with pytest.raises(AssertionError, match="independently"):
        assert_layer_conforms(Recurrent(MixesTheBatch()), torch.randn(6, 2, 3))


def test_checks_in_eval_mode_and_restores_the_mode():
    layer = Recurrent(DropsOut()).train()
    assert_layer_conforms(layer, torch.randn(6, 2, 3))
    assert layer.training and layer.cell.dropout.training


@pytest.mark.parametrize("shape", [(1, 2, 3), (6, 1, 3), (6, 3)])
def test_needs_an_input_it_can_check_with(shape):
    with pytest.raises(ValueError, match="T, B >= 2"):
        assert_layer_conforms(Recurrent(SumCell()), torch.randn(shape))


def test_echo_state_distance_decays_only_with_the_echo_state_property():
    torch.manual_seed(0)
    x = torch.randn(200, 3, 1)
    fading = echo_state_distance(reservoir(1, 32, leak=1.0, rho=0.5), x)
    assert fading.shape == (200,)
    assert fading[-1] < 1e-6 * fading[0]
    chaotic = echo_state_distance(reservoir(1, 32, leak=1.0, rho=3.0), 1e-3 * x)
    assert chaotic[-1] > 0.1 * chaotic[0]
