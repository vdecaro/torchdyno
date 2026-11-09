import pytest
import torch
from torch import nn

from tests.unit.compositions import COMPOSITIONS
from torchdyno.cells import Cell
from torchdyno.layers import Recurrent
from torchdyno.matrices import Dense
from torchdyno.testing import assert_layer_conforms


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
