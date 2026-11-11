import pytest
import torch
from torch import nn

from tests.unit.compositions import lru
from tests.unit.test_testing import SumCell
from torchdyno.layers import (
    Concat,
    RandomConv,
    Recurrent,
    Scan,
    Stack,
)
from torchdyno.pooling import (
    Last,
    Mean,
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


LAYERS = {
    "recurrent": lambda: Recurrent(SumCell()),
    "scan": lambda: Scan(lru(features=3).cell),
    "random_conv": lambda: RandomConv(3, length=12, kernels=4),
}


@pytest.mark.parametrize("make", LAYERS.values(), ids=LAYERS.keys())
@pytest.mark.parametrize("shape", [(12, 3), (0, 2, 3), (12, 2, 3, 1)])
def test_layers_reject_inputs_that_are_not_time_first_sequences(make, shape):
    with pytest.raises(ValueError, match=r"\(T, B, F\)"):
        make()(torch.randn(shape))


@pytest.mark.parametrize("make", [LAYERS["recurrent"], LAYERS["scan"]])
def test_recurrent_layers_reject_a_state_of_the_wrong_shape(make):
    layer = make()
    with pytest.raises(ValueError, match="h0"):
        layer(torch.randn(5, 2, 3), h0=torch.zeros(1, layer.cell.state_size))


def test_random_conv_rejects_another_channel_count():
    with pytest.raises(ValueError, match="3 channels"):
        RandomConv(3, length=12, kernels=4)(torch.randn(12, 2, 2))


def test_a_slice_of_a_concat_is_a_concat():
    pooled = Concat(Last(), Mean(), Last())
    first_two = pooled[:2]
    assert isinstance(first_two, Concat)
    x = torch.randn(4, 2, 3)
    torch.testing.assert_close(first_two(x), pooled(x)[..., :6])
    assert isinstance(pooled[1], Mean)
