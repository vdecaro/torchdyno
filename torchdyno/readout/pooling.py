"""Pooling: reduce a ``(T, B, H)`` sequence over time to ``(B, H)`` features.

A layer marks positions it did not produce with NaN, as
[`RandomConv`][torchdyno.sequence.RandomConv] does for kernels without padding and every
layer does at the steps a shorter sequence does not have. Every pooling skips
them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "LSPV",
    "MIPV",
    "MPV",
    "PPV",
    "Last",
    "Max",
    "Mean",
]


class Last(nn.Module):
    """The last timestep each sequence has."""

    def forward(self, x: Tensor) -> Tensor:
        """Pool over time.

        Args:
            x: the sequence, shape ``(T, B, H)``.

        Returns:
            Each sequence's last step that is not all NaN, shape ``(B, H)``;
            ``x[-1]`` for sequences of full length.
        """
        present = ~x.isnan().all(-1)  # (T, B)
        steps = torch.arange(1, len(x) + 1, device=x.device)[:, None]
        last = (present * steps).argmax(0)
        return x.gather(0, last[None, :, None].expand(1, *x.shape[1:]))[0]


class Mean(nn.Module):
    """The average over time, skipping NaN."""

    def forward(self, x: Tensor) -> Tensor:
        """Pool over time.

        Args:
            x: the sequence, shape ``(T, B, H)``.

        Returns:
            The mean over the produced positions, shape ``(B, H)``.
        """
        return x.nanmean(dim=0)


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


class MPV(nn.Module):
    """The mean of the positive values over time, as in MultiRocket."""

    def forward(self, x: Tensor) -> Tensor:
        """Pool over time.

        Args:
            x: the sequence, shape ``(T, B, H)``.

        Returns:
            The mean of the values above zero, ``0`` where there are none, shape
            ``(B, H)``.
        """
        positive = x > 0
        total = x.where(positive, 0).sum(dim=0)
        return total / positive.sum(dim=0).clamp(min=1).to(x.dtype)


class MIPV(nn.Module):
    """The mean index of the positive values, as in MultiRocket.

    Indices count the produced positions only, so they start at zero at a
    sequence's first step, whatever padding or unproduced positions precede it.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Pool over time.

        Args:
            x: the sequence, shape ``(T, B, H)``.

        Returns:
            The mean index of the values above zero, ``-1`` where there are none,
            shape ``(B, H)``.
        """
        positive = x > 0
        index = (~x.isnan()).cumsum(dim=0) - 1
        count = positive.sum(dim=0)
        mean = index.where(positive, 0).sum(dim=0).to(x.dtype) / count.clamp(min=1)
        return mean.where(count > 0, -1)


class LSPV(nn.Module):
    """The longest stretch of consecutive positive values, as in MultiRocket."""

    def forward(self, x: Tensor) -> Tensor:
        """Pool over time.

        Args:
            x: the sequence, shape ``(T, B, H)``.

        Returns:
            The length of the longest run of values above zero, shape
            ``(B, H)``.
        """
        positive = x > 0
        count = positive.cumsum(dim=0)
        # a run's length is the count since the last position that was not positive
        before = count.where(~positive, 0).cummax(dim=0).values
        return (count - before).amax(dim=0).to(x.dtype)
