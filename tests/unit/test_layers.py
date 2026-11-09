import torch

from tests.unit.test_testing import SumCell
from torchdyno.layers import Recurrent


def test_recurrent_starts_from_zeros():
    layer = Recurrent(SumCell())
    x = torch.randn(4, 2, 3)
    torch.testing.assert_close(layer(x), layer.cell.input(x).cumsum(0))
