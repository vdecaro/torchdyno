"""Maps along time without a state of their own: reversal and differencing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


class Reverse(nn.Module):
    """Run a layer from the last timestep to the first.

    The output is aligned with the input, so ``Concat(layer, Reverse(other))``
    reads a sequence in both directions. Steps padded with NaN at the end of a
    shorter sequence come first in reverse, where a recurrent layer holds its
    initial state over them.
    """

    def __init__(self, layer: nn.Module):
        """Initializes the layer.

        Args:
            layer: the ``(T, B, F)`` to ``(T, B, H)`` layer to run in reverse.
        """
        super().__init__()
        self.layer = layer

    def forward(self, x: Tensor) -> Tensor:
        """Run the layer over ``x`` reversed in time.

        Args:
            x: the input sequence, shape ``(T, B, F)``.

        Returns:
            The layer's output, flipped back to the input's order.
        """
        return self.layer(x.flip(0)).flip(0)


class Diff(nn.Module):
    """The first difference over time: ``x_t - x_(t-1)``.

    The first step has no predecessor and is NaN, so it is absent for the layers
    and the pooling that follow. Concatenated with the input it would be partly
    NaN, which the learners reject; keep both by differencing after a layer.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Difference ``x`` over time.

        Args:
            x: the sequence, shape ``(T, B, F)``.

        Returns:
            The differences, same shape, NaN at the first step.
        """
        return torch.cat([torch.full_like(x[:1], torch.nan), x.diff(dim=0)])
