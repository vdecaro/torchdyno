"""Combining sequence layers: in series, or side by side."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Self,
    cast,
    overload,
)

import torch
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor

    from torchdyno.sequence.recurrent import Recurrent


class Stack(nn.Sequential):
    """Run layers in series and concatenate every layer's output.

    Each layer reads the previous layer's output; the result is all of them
    side by side, e.g. a deep reservoir read out over every layer. With
    ``feed_input=True`` every layer after the first also reads the input,
    ``[previous output; x]``.
    """

    def __init__(self, *layers: nn.Module, feed_input: bool = False):
        """Initializes the stack.

        Args:
            layers: the layers, in order.
            feed_input: give every layer after the first the input as well.
        """
        super().__init__(*layers)
        self.feed_input = feed_input

    def forward(self, x: Tensor, h0: Tensor | None = None) -> Tensor:
        """Run the layers.

        Args:
            x: the input sequence, shape ``(T, B, F)``.
            h0: the layers' states to start from, side by side, e.g. the last
                output of a previous call; ``None`` starts every layer afresh.

        Returns:
            The outputs of all layers, concatenated on the last dimension.
        """
        starts: list[Tensor | None] = [None] * len(self)
        if h0 is not None:
            sizes = [cast("Recurrent", layer).cell.state_size for layer in self]
            starts = list(h0.split(sizes, -1))
        outputs: list[Tensor] = []
        for layer, start in zip(self, starts, strict=True):
            inputs = x if not outputs else outputs[-1]
            if outputs and self.feed_input:
                inputs = torch.cat([inputs, x], dim=-1)
            outputs.append(layer(inputs) if start is None else layer(inputs, h0=start))
        return torch.cat(outputs, dim=-1)


class Concat(nn.ModuleList):
    """Run modules side by side on the same input and concatenate their outputs.

    E.g. ``Concat(PPV(), Max())`` pools a sequence two ways at once, and
    ``Concat(layer, Reverse(other))`` reads it in both directions.
    """

    def __init__(self, *modules: nn.Module):
        """Initializes the layer.

        Args:
            modules: the modules to run.
        """
        super().__init__(modules)

    @overload
    def __getitem__(self, idx: slice) -> Self: ...

    @overload
    def __getitem__(self, idx: int) -> nn.Module: ...

    def __getitem__(self, idx: int | slice) -> nn.Module:
        """A module, or the same combination of a slice of them."""
        if isinstance(idx, slice):
            return type(self)(*list(self)[idx])
        return super().__getitem__(idx)

    def forward(self, x: Tensor) -> Tensor:
        """Run every module on ``x``.

        Args:
            x: the input.

        Returns:
            The outputs, concatenated on the last dimension.
        """
        return torch.cat([module(x) for module in self], dim=-1)


class Sum(Concat):
    """Run modules side by side on the same input and add their outputs.

    E.g. ``Sum(nn.Identity(), layer)`` is a residual connection around
    ``layer``.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Run every module on ``x``.

        Args:
            x: the input.

        Returns:
            The sum of the outputs, which must have the same shape.
        """
        return torch.stack([module(x) for module in self]).sum(0)
