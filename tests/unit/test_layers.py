import torch
from torch import nn

from tests.unit.test_testing import SumCell
from torchdyno.layers import (
    Recurrent,
    Stack,
)
from torchdyno.testing import assert_layer_conforms


def test_recurrent_starts_from_zeros():
    layer = Recurrent(SumCell())
    x = torch.randn(4, 2, 3)
    torch.testing.assert_close(layer(x), layer.cell.input(x).cumsum(0))


def test_stack_concatenates_layers_run_in_series():
    first, second = Recurrent(SumCell()), Recurrent(SumCell())
    x = torch.randn(4, 2, 3)
    out = Stack(first, second)(x)
    assert out.shape == (4, 2, 6)
    torch.testing.assert_close(out[..., :3], first(x))
    torch.testing.assert_close(out[..., 3:], nn.Sequential(first, second)(x))
    assert_layer_conforms(first, x)
