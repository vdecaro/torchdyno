import torch

from torchdyno.layers import Concat
from torchdyno.pooling import (
    PPV,
    Last,
    Max,
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


def test_max_and_ppv_skip_positions_marked_as_missing():
    x = torch.tensor(
        [[[float("nan"), 3.0]], [[-1.0, -2.0]], [[2.0, 1.0]]]
    )  # (T=3, B=1, H=2)
    torch.testing.assert_close(Max()(x), torch.tensor([[2.0, 3.0]]))
    torch.testing.assert_close(PPV()(x), torch.tensor([[1 / 2, 2 / 3]]))
    torch.testing.assert_close(Mean()(x), torch.tensor([[0.5, 2 / 3]]))


def test_max_and_ppv_without_missing_positions():
    x = torch.randn(7, 3, 4)
    torch.testing.assert_close(Max()(x), x.amax(0))
    torch.testing.assert_close(PPV()(x), (x > 0).float().mean(0))
