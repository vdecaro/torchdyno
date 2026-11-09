"""Pooling: reduce a ``(T, B, H)`` sequence over time to ``(B, H)`` features."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from torch import Tensor


class Last(nn.Module):
    """The last timestep."""

    def forward(self, x: Tensor) -> Tensor:
        """Pool over time.

        Args:
            x: the sequence, shape ``(T, B, H)``.

        Returns:
            ``x[-1]``, shape ``(B, H)``.
        """
        return x[-1]


class Mean(nn.Module):
    """The average over time."""

    def forward(self, x: Tensor) -> Tensor:
        """Pool over time.

        Args:
            x: the sequence, shape ``(T, B, H)``.

        Returns:
            The mean over time, shape ``(B, H)``.
        """
        return x.mean(dim=0)
