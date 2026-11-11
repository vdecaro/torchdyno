"""Pooling: reduce a ``(T, B, H)`` sequence over time to ``(B, H)`` features.

A layer marks positions it did not produce with NaN, as
:class:`~torchdyno.sequence.RandomConv` does for kernels without padding and every
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
