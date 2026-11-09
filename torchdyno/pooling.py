"""Pooling: reduce a ``(T, B, H)`` sequence over time to ``(B, H)`` features.

A layer may mark positions it did not produce with NaN.
:class:`Max` and :class:`PPV` skip them; :class:`Last` and :class:`Mean`
propagate them, so a misuse stays visible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
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


class Max(nn.Module):
    """The maximum over time, skipping NaN."""

    def forward(self, x: Tensor) -> Tensor:
        """Pool over time.

        Args:
            x: the sequence, shape ``(T, B, H)``.

        Returns:
            The maximum over the produced positions, shape ``(B, H)``.
        """
        return x.masked_fill(x.isnan(), -torch.inf).amax(dim=0)


class PPV(nn.Module):
    """The proportion of positive values over time, skipping NaN."""

    def forward(self, x: Tensor) -> Tensor:
        """Pool over time.

        Args:
            x: the sequence, shape ``(T, B, H)``.

        Returns:
            The fraction of produced positions above zero, shape ``(B, H)``.
        """
        produced = (~x.isnan()).sum(dim=0)
        return (x > 0).sum(dim=0).to(x.dtype) / produced.to(x.dtype)
