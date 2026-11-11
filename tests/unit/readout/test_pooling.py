import pytest
import torch

from torchdyno.readout import (
    LSPV,
    MIPV,
    MPV,
    PPV,
    Last,
    Max,
    Mean,
)
from torchdyno.sequence import Concat


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


def multirocket(series):
    """MPV, MIPV and LSPV of one series, as MultiRocket loops over it."""
    values = series[~series.isnan()].tolist()
    positive = [i for i, v in enumerate(values) if v > 0]
    longest = run = 0
    for v in values:
        run = run + 1 if v > 0 else 0
        longest = max(longest, run)
    mpv = sum(values[i] for i in positive) / len(positive) if positive else 0.0
    mipv = sum(positive) / len(positive) if positive else -1.0
    return mpv, mipv, float(longest)


def test_multirocket_pooling_matches_a_loop_and_skips_unproduced_positions():
    torch.manual_seed(0)
    x = torch.randn(12, 3, 4, dtype=torch.float64)
    x[:3, 0] = torch.nan  # unproduced leading positions
    x[:, 1, 2] = -1.0  # never positive
    pooled = [pool(x) for pool in (MPV(), MIPV(), LSPV())]
    for b in range(3):
        for h in range(4):
            expected = multirocket(x[:, b, h])
            assert tuple(p[b, h].item() for p in pooled) == pytest.approx(expected)
