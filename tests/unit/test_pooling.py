import torch

from torchdyno.layers import Concat
from torchdyno.pooling import (
    Last,
    Mean,
)


def test_pooling_over_time():
    x = torch.randn(5, 3, 4)
    assert torch.equal(Last()(x), x[-1])
    torch.testing.assert_close(Mean()(x), x.mean(0))


def test_concat_runs_modules_side_by_side():
    x = torch.randn(5, 3, 4)
    out = Concat(Last(), Mean())(x)
    assert out.shape == (3, 8)
    assert torch.equal(out[:, :4], x[-1])
