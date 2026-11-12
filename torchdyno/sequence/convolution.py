"""Fixed random convolution kernels (ROCKET, MiniRocket)."""

from __future__ import annotations

import itertools
import math
from typing import (
    TYPE_CHECKING,
    NamedTuple,
)

import torch
import torch.nn.functional as F
from torch import nn

from torchdyno.sequence._checks import (
    _absent,
    _check,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor


class Kernels(NamedTuple):
    """A bank of 1D convolution kernels, as ``RandomConv`` holds them.

    Attributes:
        weight: ``(K, C, L)`` weights, zero-padded past each kernel's length.
        bias: ``(K,)`` biases.
        lengths: ``(K,)`` kernel lengths, at most ``L``.
        dilations: ``(K,)`` dilations.
        paddings: ``(K,)`` paddings on each side.
    """

    weight: Tensor
    bias: Tensor
    lengths: Tensor
    dilations: Tensor
    paddings: Tensor


def rocket(
    channels: int, length: int, kernels: int, dtype: torch.dtype | None = None
) -> Kernels:
    """Draw ROCKET's random kernels.

    From Dempster, Petitjean and Webb, "ROCKET: exceptionally fast and accurate
    time series classification using random convolutional kernels" (2020), with
    the multivariate kernels of its reference implementations. Each kernel has:

    - a length drawn from those of {7, 9, 11} that fit the series;
    - N(0, 1) weights, centred, on each of a random subset of channels;
    - a U(-1, 1) bias;
    - a dilation ``int(2 ** U(0, log2((length - 1) / (L - 1))))``;
    - padding ``(L - 1) d // 2`` half of the time.

    Args:
        channels: the number of input channels.
        length: the series length the kernels are drawn for.
        kernels: the number of kernels.
        dtype: the data type of the weights and biases; ``None`` takes
            torch's default.

    Returns:
        The kernels.

    Raises:
        ValueError: if ``length`` is shorter than the shortest kernel.
    """
    dtype = dtype or torch.get_default_dtype()
    sizes = [size for size in (7, 9, 11) if size <= length]
    if not sizes:
        raise ValueError(f"length must be at least 7, the kernel minimum: {length}")
    lengths = torch.tensor(sizes)[torch.randint(len(sizes), (kernels,))]
    weight = torch.zeros(kernels, channels, 11, dtype=dtype)
    dilations = torch.ones(kernels, dtype=torch.long)
    paddings = torch.zeros(kernels, dtype=torch.long)
    for k, size in enumerate(lengths.tolist()):
        used = int(2 ** (torch.rand(()).item() * math.log2(min(channels, size) + 1)))
        taps = torch.randn(used, size, dtype=dtype)
        weight[k, torch.randperm(channels)[:used], :size] = taps - taps.mean(
            dim=1, keepdim=True
        )
        spread = math.log2((length - 1) / (size - 1))
        dilations[k] = max(1, int(2 ** (torch.rand(()).item() * spread)))
        if torch.rand(()).item() < 0.5:  # noqa: PLR2004
            paddings[k] = (size - 1) * int(dilations[k]) // 2
    bias = torch.empty(kernels, dtype=dtype).uniform_(-1, 1)
    return Kernels(weight, bias, lengths, dilations, paddings)


def minirocket(
    channels: int, length: int, kernels: int, dtype: torch.dtype | None = None
) -> Kernels:
    """Draw MiniRocket's kernels: 84 fixed patterns at spread dilations.

    From Dempster, Schmidt and Webb, "MiniRocket: a very fast (almost)
    deterministic transform for time series classification" (2021). Each of the
    84 length-9 patterns has weight 2 at three taps and -1 at the other six, on
    a random subset of channels. Dilations are ``2 ** linspace(0, log2((length
    - 1) / 8))``, up to 32 of them; each (dilation, pattern) pair is repeated
    for as many kernels as its share of ``kernels``, and is padded for every
    other pair. Rows run over dilations, then patterns, then repeats. Biases are
    zero: fit them with [`QuantileBias`][torchdyno.learners.QuantileBias], and pool with
    [`PPV`][torchdyno.readout.PPV].

    Args:
        channels: the number of input channels.
        length: the series length the kernels are drawn for.
        kernels: the number of kernels, a multiple of 84, e.g. 9,996.
        dtype: the data type of the weights and biases; ``None`` takes
            torch's default.

    Returns:
        The kernels.

    Raises:
        ValueError: if ``length`` is shorter than 9 or ``kernels`` is not a
            multiple of 84.
    """
    dtype = dtype or torch.get_default_dtype()
    patterns = list(itertools.combinations(range(9), 3))
    if length < 9:  # noqa: PLR2004
        raise ValueError(f"length must be at least 9, the kernel length: {length}")
    if kernels % len(patterns):
        raise ValueError(f"MiniRocket needs a multiple of 84 kernels, got {kernels}.")
    repeats = kernels // len(patterns)
    steps = min(repeats, 32)
    spread = torch.linspace(0, math.log2((length - 1) / 8), steps, dtype=torch.float64)
    dilations, counts = torch.unique((2**spread).long(), return_counts=True)
    counts = (counts * repeats / steps).long()
    counts[: repeats - int(counts.sum())] += 1
    rows: list[tuple[list[int], int, int, int]] = []
    for i, (dilation, count) in enumerate(
        zip(dilations.tolist(), counts.tolist(), strict=True)
    ):
        for j in range(len(patterns)):
            used = int(2 ** (torch.rand(()).item() * math.log2(min(channels, 9) + 1)))
            subset = torch.randperm(channels)[:used].tolist()
            padding = 4 * dilation if (i + j) % 2 == 0 else 0
            rows += [(subset, j, dilation, padding)] * count
    taps = torch.full((len(patterns), 9), -1.0, dtype=dtype)
    for j, pattern in enumerate(patterns):
        taps[j, list(pattern)] = 2.0
    weight = torch.zeros(kernels, channels, 9, dtype=dtype)
    for k, (subset, j, _, _) in enumerate(rows):
        weight[k, subset] = taps[j]
    return Kernels(
        weight,
        torch.zeros(kernels, dtype=dtype),
        torch.full((kernels,), 9),
        torch.tensor([row[2] for row in rows]),
        torch.tensor([row[3] for row in rows]),
    )


class RandomConv(nn.Module):
    """Fixed random convolution kernels: ``(T, B, C) -> (T, B, kernels)``.

    The kernels come from ``draw``, [`rocket()`][torchdyno.sequence.rocket] by default,
    the way a matrix takes an initialiser, and are kept as buffers. A kernel without
    padding produces ``T - (L - 1) d`` outputs, aligned to the end, and the positions
    before them are NaN: pool with [`PPV`][torchdyno.readout.PPV] and
    [`Max`][torchdyno.readout.Max], which skip them.
    """

    weight: Tensor
    bias: Tensor
    lengths: Tensor
    dilations: Tensor
    paddings: Tensor

    def __init__(
        self,
        channels: int,
        length: int,
        kernels: int = 10_000,
        dtype: torch.dtype | None = None,
        draw: Callable[[int, int, int, torch.dtype], Kernels] = rocket,
    ):
        """Initializes the layer.

        Args:
            channels: the number of input channels ``C``.
            length: the series length the kernels are drawn for; inputs must be
                at least this long.
            kernels: the number of kernels.
            dtype: the data type; ``None`` takes torch's default.
            draw: ``draw(channels, length, kernels, dtype) -> Kernels``.

        Raises:
            ValueError: if ``channels`` or ``kernels`` is not positive, or the
                drawn kernels do not have ``kernels`` rows of ``channels``.
        """
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        if channels < 1 or kernels < 1:
            raise ValueError(f"Need channels, kernels >= 1, got {channels}, {kernels}.")
        bank = draw(channels, length, kernels, dtype)
        if bank.weight.shape[:2] != (kernels, channels):
            raise ValueError(
                f"draw gave {tuple(bank.weight.shape[:2])} kernels by channels, "
                f"not {(kernels, channels)}."
            )
        self.channels = channels
        self.length = length
        self.kernels = kernels
        for name, value in bank._asdict().items():
            self.register_buffer(name, value)

    def forward(self, x: Tensor) -> Tensor:
        """Convolve every kernel with ``x``.

        Args:
            x: the input sequence, shape ``(T, B, C)``.

        Returns:
            The kernel outputs, shape ``(T, B, kernels)``, NaN where a kernel
            without padding produced nothing.

        Raises:
            ValueError: if ``x`` is not a ``(T, B, channels)`` sequence at least
                ``length`` long.
        """
        _check(x)
        if x.shape[-1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[-1]}.")
        if x.shape[0] < self.length:
            raise ValueError(
                f"The kernels were drawn for series of at least {self.length} steps, "
                f"got {x.shape[0]}."
            )
        steps = x.shape[0]
        absent = _absent(x).T[:, None]  # (B, 1, T)
        signal = x.permute(1, 2, 0).masked_fill(absent, 0)  # conv1d wants (B, C, T)
        out = signal.new_full((x.shape[1], self.kernels, steps), torch.nan)
        groups = torch.stack([self.lengths, self.dilations, self.paddings], dim=1)
        for group in groups.unique(dim=0):
            index = (groups == group).all(dim=1).nonzero().squeeze(1)
            size, dilation, padding = group.tolist()
            conv = F.conv1d(
                signal,
                self.weight[index, :, :size],
                self.bias[index],
                padding=padding,
                dilation=dilation,
            )
            if absent.any():
                conv = conv.masked_fill(
                    _unproduced(absent, size, dilation, padding), torch.nan
                )
            out[:, index, steps - conv.shape[-1] :] = conv
        return out.permute(2, 0, 1)


def _unproduced(absent: Tensor, size: int, dilation: int, padding: int) -> Tensor:
    """Where a kernel's output depends on steps a sequence does not have.

    A padded kernel's output stands for the step at its centre, which zero
    padding past a shorter sequence's end does not change; an unpadded one
    needs its whole window.
    """
    if padding:
        return absent
    ones = absent.new_ones(1, 1, size, dtype=torch.float32)
    return F.conv1d(absent.float(), ones, dilation=dilation) > 0
