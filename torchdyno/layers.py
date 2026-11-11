"""Layers: modules that map a ``(T, B, F)`` sequence to a ``(T, B, H)`` one."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor

    from torchdyno.cells import (
        Cell,
        LinearDiagonalCell,
    )


def _check(x: Tensor, h0: Tensor | None = None, state_size: int = 0) -> None:
    """Raise unless ``x`` is a ``(T, B, F)`` sequence and ``h0`` a ``(B, H)`` state."""
    if x.ndim != 3 or x.shape[0] == 0:  # noqa: PLR2004
        raise ValueError(
            f"Expected a time-first (T, B, F) sequence with T >= 1, got shape "
            f"{tuple(x.shape)}; add a batch dimension with x.unsqueeze(1)."
        )
    if h0 is not None and h0.shape != (x.shape[1], state_size):
        raise ValueError(
            f"h0 must be (B, state_size) = {(x.shape[1], state_size)}, got "
            f"{tuple(h0.shape)}."
        )


class Recurrent(nn.Module):
    """Run a cell over time, one step after the other."""

    def __init__(self, cell: Cell):
        """Initializes the layer.

        Args:
            cell: the cell to run.
        """
        super().__init__()
        self.cell = cell

    def forward(self, x: Tensor, h0: Tensor | None = None) -> Tensor:
        """Run the cell over ``x``.

        Args:
            x: the input sequence, shape ``(T, B, F)``.
            h0: the state to start from, e.g. the last output of a previous
                call; ``None`` starts from zeros.

        Returns:
            The state after every timestep, shape ``(T, B, state_size)``.

        Raises:
            ValueError: if ``x`` or ``h0`` has the wrong shape.
        """
        _check(x, h0, self.cell.state_size)
        h = x.new_zeros(x.shape[1], self.cell.state_size) if h0 is None else h0
        states = []
        for u in x:
            h = self.cell(u, h)
            states.append(h)
        return torch.stack(states)


class Scan(Recurrent):
    """Run a :class:`~torchdyno.cells.LinearDiagonalCell` with a parallel scan.

    Same contract as :class:`Recurrent`, but ``ceil(log2 T)`` steps over the whole
    sequence replace the ``T`` sequential ones, which the linear recurrence
    allows.
    """

    cell: LinearDiagonalCell

    def __init__(self, cell: LinearDiagonalCell):
        """Initializes the layer.

        Args:
            cell: the linear cell to run.
        """
        super().__init__(cell)

    def forward(self, x: Tensor, h0: Tensor | None = None) -> Tensor:
        """Run the cell over ``x``.

        Args:
            x: the input sequence, shape ``(T, B, F)``.
            h0: the state to start from, e.g. the last output of a previous
                call; ``None`` starts from zeros.

        Returns:
            The state after every timestep, shape ``(T, B, state_size)``.

        Raises:
            ValueError: if ``x`` or ``h0`` has the wrong shape.
        """
        _check(x, h0, self.cell.state_size)
        a, b = self.cell.coefficients(x)
        if h0 is not None:
            modes = self.cell.modes
            first = b[0] + a * torch.complex(h0[..., :modes], h0[..., modes:])
            b = torch.cat([first.unsqueeze(0), b[1:]])
        h = _scan(a.expand_as(b), b)
        return torch.cat([h.real, h.imag], dim=-1)


def _scan(a: Tensor, b: Tensor) -> Tensor:
    """Hillis-Steele scan of ``h_t = a_t h_(t-1) + b_t`` over dim 0, from zero."""
    shift = 1
    while shift < b.shape[0]:
        a, b = (
            torch.cat([a[:shift], a[shift:] * a[:-shift]]),
            torch.cat([b[:shift], a[shift:] * b[:-shift] + b[shift:]]),
        )
        shift *= 2
    return b


class Stack(nn.Sequential):
    """Run layers in series and concatenate every layer's output.

    Each layer reads the previous layer's output; the result is all of them
    side by side, e.g. a deep reservoir read out over every layer.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Run the layers.

        Args:
            x: the input sequence, shape ``(T, B, F)``.

        Returns:
            The outputs of all layers, concatenated on the last dimension.
        """
        outputs = []
        for layer in self:
            x = layer(x)
            outputs.append(x)
        return torch.cat(outputs, dim=-1)


class Concat(nn.ModuleList):
    """Run modules side by side on the same input and concatenate their outputs.

    E.g. ``Concat(PPV(), Max())`` pools a sequence two ways at once.
    """

    def __init__(self, *modules: nn.Module):
        """Initializes the layer.

        Args:
            modules: the modules to run.
        """
        super().__init__(modules)

    def forward(self, x: Tensor) -> Tensor:
        """Run every module on ``x``.

        Args:
            x: the input.

        Returns:
            The outputs, concatenated on the last dimension.
        """
        return torch.cat([module(x) for module in self], dim=-1)


class RandomConv(nn.Module):
    """ROCKET's random convolution kernels: ``(T, B, C) -> (T, B, kernels)``.

    From Dempster, Petitjean and Webb, "ROCKET: exceptionally fast and accurate
    time series classification using random convolutional kernels" (2020), with
    the multivariate kernels of its reference implementations. Each kernel has:

    - a length drawn from those of {7, 9, 11} that fit the series;
    - N(0, 1) weights, centred on each of a random subset of channels;
    - a U(-1, 1) bias;
    - a dilation ``int(2 ** U(0, log2((length - 1) / (L - 1))))``;
    - padding ``(L - 1) d // 2`` half of the time.

    The kernels are fixed buffers. A kernel without padding produces
    ``T - (L - 1) d`` outputs, aligned to the end, and the positions before them
    are NaN: pool with :class:`~torchdyno.pooling.PPV` and
    :class:`~torchdyno.pooling.Max`, which skip them.
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
        dtype: torch.dtype = torch.float32,
    ):
        """Initializes the layer.

        Args:
            channels: the number of input channels ``C``.
            length: the series length the kernels are drawn for; inputs must be
                at least this long.
            kernels: the number of kernels.
            dtype: the data type.

        Raises:
            ValueError: if ``channels`` or ``kernels`` is not positive, or
                ``length`` is shorter than the shortest kernel.
        """
        super().__init__()
        if channels < 1 or kernels < 1:
            raise ValueError(f"Need channels, kernels >= 1, got {channels}, {kernels}.")
        sizes = [size for size in (7, 9, 11) if size <= length]
        if not sizes:
            raise ValueError(f"length must be at least 7, the kernel minimum: {length}")
        self.channels = channels
        self.length = length
        self.kernels = kernels
        lengths = torch.tensor(sizes)[torch.randint(len(sizes), (kernels,))]
        weight = torch.zeros(kernels, channels, 11, dtype=dtype)
        dilations = torch.ones(kernels, dtype=torch.long)
        paddings = torch.zeros(kernels, dtype=torch.long)
        for k, size in enumerate(lengths.tolist()):
            used = int(
                2 ** (torch.rand(()).item() * math.log2(min(channels, size) + 1))
            )
            taps = torch.randn(used, size, dtype=dtype)
            weight[k, torch.randperm(channels)[:used], :size] = taps - taps.mean(
                dim=1, keepdim=True
            )
            spread = math.log2((length - 1) / (size - 1))
            dilations[k] = max(1, int(2 ** (torch.rand(()).item() * spread)))
            if torch.rand(()).item() < 0.5:  # noqa: PLR2004
                paddings[k] = (size - 1) * int(dilations[k]) // 2
        self.register_buffer("weight", weight)
        self.register_buffer("bias", torch.empty(kernels, dtype=dtype).uniform_(-1, 1))
        self.register_buffer("lengths", lengths)
        self.register_buffer("dilations", dilations)
        self.register_buffer("paddings", paddings)

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
        signal = x.permute(1, 2, 0)  # conv1d wants (B, C, T)
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
            out[:, index, steps - conv.shape[-1] :] = conv
        return out.permute(2, 0, 1)
