import math

import pytest
import torch
from torch import nn

from torchdyno.learners import Ridge
from torchdyno.readout import (
    PPV,
    Max,
)
from torchdyno.sequence import (
    Concat,
    Kernels,
    RandomConv,
    minirocket,
    rocket,
)


def _apply_kernels(conv, x):
    """The reference ``apply_kernel`` loop, keeping every output of every kernel."""
    steps, batch, _ = x.shape
    out = torch.full((steps, batch, conv.kernels), torch.nan, dtype=x.dtype)
    for k in range(conv.kernels):
        size, dilation, padding = (
            int(v[k]) for v in (conv.lengths, conv.dilations, conv.paddings)
        )
        produced = steps + 2 * padding - (size - 1) * dilation
        for i in range(-padding, steps + padding - (size - 1) * dilation):
            total = conv.bias[k].expand(batch).clone()
            for j in range(size):
                index = i + j * dilation
                if 0 <= index < steps:
                    total += x[index] @ conv.weight[k, :, j]
            out[steps - produced + i + padding, :, k] = total
    return out


def test_matches_the_reference_kernel_loop():
    for channels in (1, 3):
        torch.manual_seed(channels)
        conv = RandomConv(channels, length=40, kernels=24, dtype=torch.float64)
        assert conv.paddings.eq(0).any() and conv.paddings.gt(0).any()
        assert conv.dilations.gt(1).any()
        x = torch.randn(40, 2, channels, dtype=torch.float64)
        torch.testing.assert_close(conv(x), _apply_kernels(conv, x), equal_nan=True)


def test_kernels_are_drawn_as_in_rocket():
    torch.manual_seed(0)
    channels, length, kernels = 4, 100, 2000
    conv = RandomConv(channels, length, kernels)
    sizes, dilations, paddings = conv.lengths, conv.dilations, conv.paddings
    assert set(sizes.tolist()) == {7, 9, 11}
    assert (conv.bias.abs() <= 1).all()
    assert (dilations >= 1).all() and ((sizes - 1) * dilations <= length - 1).all()
    padded = paddings > 0
    assert torch.equal(paddings[padded], (sizes - 1)[padded] * dilations[padded] // 2)
    assert 0.45 < padded.float().mean() < 0.55
    for k in range(kernels):
        size = int(sizes[k])
        assert not conv.weight[k, :, size:].any()
        used = conv.weight[k, :, :size].ne(0).any(dim=1)
        assert 1 <= used.sum() <= min(channels, size)
        torch.testing.assert_close(
            conv.weight[k, used, :size].sum(1), torch.zeros(int(used.sum()))
        )
    assert RandomConv(1, length, 200).weight[:, 0].ne(0).any(dim=1).all()
    assert math.isclose(
        dilations.float().log2().max().item(), math.log2(99 / 6), abs_tol=1.0
    )


def test_only_unpadded_kernels_leave_leading_positions_missing():
    torch.manual_seed(0)
    conv = RandomConv(2, length=60, kernels=50)
    missing = conv(torch.randn(60, 3, 2)).isnan()
    expected = torch.where(conv.paddings > 0, 0, (conv.lengths - 1) * conv.dilations)
    assert torch.equal(missing[:, 0].sum(0), expected)
    for k in range(50):
        assert missing[: int(expected[k]), :, k].all()


def test_kernels_are_fixed_and_saved_with_the_model():
    torch.manual_seed(0)
    conv = RandomConv(2, length=30, kernels=20)
    assert not list(conv.parameters())
    torch.manual_seed(1)
    copy = RandomConv(2, length=30, kernels=20)
    copy.load_state_dict(conv.state_dict())
    x = torch.randn(30, 4, 2)
    torch.testing.assert_close(copy(x), conv(x), equal_nan=True)


def test_features_separate_two_kinds_of_series():
    torch.manual_seed(0)
    t = torch.linspace(0, 8 * math.pi, 80)
    phases = torch.rand(120) * 2 * math.pi
    labels = torch.arange(120) % 2
    frequency = 1.0 + labels.float()  # class 1 oscillates twice as fast
    x = torch.sin(frequency[None] * t[:, None] + phases[None]).unsqueeze(-1)
    x = x + 0.3 * torch.randn_like(x)
    model = nn.Sequential(
        RandomConv(1, 80, 500), Concat(PPV(), Max()), nn.Linear(1000, 2)
    )
    Ridge(l2=1.0, standardize=True).fit(model, [(x[:, :80], labels[:80])])
    accuracy = (model(x[:, 80:]).argmax(-1) == labels[80:]).float().mean()
    assert accuracy > 0.9


def test_short_series_get_only_kernels_that_fit():
    layer = RandomConv(1, length=8, kernels=50)
    assert set(layer.lengths.tolist()) == {7}
    assert layer(torch.randn(8, 2, 1)).shape == (8, 2, 50)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"channels": 0, "length": 20}, "channels"),
        ({"channels": 1, "length": 20, "kernels": 0}, "kernels"),
        ({"channels": 1, "length": 6}, "at least 7"),
    ],
)
def test_rejects_sizes_it_cannot_draw_for(kwargs, match):
    with pytest.raises(ValueError, match=match):
        RandomConv(**kwargs)


def test_rejects_series_shorter_than_drawn_for():
    with pytest.raises(ValueError, match="at least 20 steps"):
        RandomConv(1, length=20, kernels=4)(torch.randn(19, 2, 1))


def test_takes_its_kernels_from_any_draw():
    def fixed(channels, length, kernels, dtype):  # noqa: ARG001
        weight = torch.zeros(kernels, channels, 3, dtype=dtype)
        weight[:, :, 1] = 1.0  # identity kernels
        ones, zeros = torch.ones(kernels, dtype=torch.long), torch.zeros(kernels)
        return Kernels(weight, zeros.to(dtype), 3 * ones, ones, ones)

    x = torch.randn(10, 2, 1)
    out = RandomConv(1, length=10, kernels=4, draw=fixed)(x)
    torch.testing.assert_close(out, x.expand(-1, -1, 4))


def test_rejects_a_draw_of_another_shape():
    def wrong(channels, length, kernels, dtype):
        return rocket(channels, length, kernels + 1, dtype)

    with pytest.raises(ValueError, match="kernels by channels"):
        RandomConv(1, length=20, kernels=4, draw=wrong)


def test_minirocket_draws_84_patterns_at_the_reference_dilations():
    torch.manual_seed(0)
    conv = RandomConv(3, length=60, kernels=84 * 119, draw=minirocket)
    weights = conv.weight[conv.weight.abs().sum(-1) > 0]  # the channels in use
    assert ((weights == 2).sum(-1) == 3).all() and ((weights == -1).sum(-1) == 6).all()
    dilations, counts = conv.dilations.unique(return_counts=True)
    assert dilations.tolist() == list(range(1, 8))
    # numpy's logspace and int32 counts in the reference fit_dilations
    assert (counts // 84).tolist() == [41, 27, 15, 11, 11, 11, 3]
    first = torch.cat([torch.tensor([0]), counts.cumsum(0)[:-1]])
    for i, (start, count) in enumerate(
        zip(first.tolist(), (counts // 84).tolist(), strict=True)
    ):
        padded = conv.paddings[start : start + 84 * count : count] > 0
        assert padded.tolist() == [(i + j) % 2 == 0 for j in range(84)]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"length": 8, "kernels": 84}, "at least 9"),
        ({"length": 20, "kernels": 100}, "84"),
    ],
)
def test_minirocket_rejects_sizes_it_cannot_draw_for(kwargs, match):
    with pytest.raises(ValueError, match=match):
        RandomConv(1, draw=minirocket, **kwargs)
