"""RLS: fit the final linear readout online, by recursive least squares."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from torchdyno.learners.base import (
    FitResult,
    feature_rows,
)
from torchdyno.reproducibility import get_rng_state

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor


class RLS:
    """Fit ``model[-1]``, an ``nn.Linear``, one sample at a time.

    Rows are visited in order: every timestep of every sequence for
    time-aligned features, one row per sequence for pooled ones. Starting from
    ``W = 0`` and ``P = I / l2``, one pass gives the ridge solution with penalty
    ``l2``; a bias is fitted as the weight of a constant feature, so unlike
    :class:`~torchdyno.learners.Ridge` it is penalised too. Each update costs
    ``O(H²)``.
    """

    def __init__(self, l2: float = 1.0, washout: int = 0):
        """Initializes the learner.

        Args:
            l2: the initial ``P = I / l2``, i.e. the ridge penalty.
            washout: leading timesteps to discard; only for time-aligned
                features.
        """
        self.l2 = l2
        self.washout = washout

    def fit(
        self,
        model: nn.Sequential,
        train: Iterable[tuple[Tensor, Tensor]],
        val: Iterable[tuple[Tensor, Tensor]] | None = None,  # noqa: ARG002
    ) -> FitResult:
        """Update the readout over every row of ``train``.

        Args:
            model: the model whose last module is set in place.
            train: batches of time-first ``(x, y)``, as for
                :class:`~torchdyno.learners.Ridge`.
            val: unused; accepted so every learner fits the same way.

        Returns:
            The RNG state at the start.

        Raises:
            TypeError: if the last module is not an ``nn.Linear``.
        """
        readout = model[-1]
        if not isinstance(readout, nn.Linear):
            raise TypeError(f"The last module must be nn.Linear, not {type(readout)}.")
        rng = get_rng_state()
        bias = readout.bias is not None
        size = readout.in_features + bias
        device = readout.weight.device
        p = torch.eye(size, dtype=torch.float64, device=device) / self.l2
        w = torch.zeros(size, readout.out_features, dtype=torch.float64, device=device)
        for x, y in train:
            rows, targets = feature_rows(model, x, y, self.washout)
            if bias:
                rows = torch.cat([rows, rows.new_ones(len(rows), 1)], dim=1)
            for row, target in zip(rows, targets, strict=True):
                p_row = p @ row
                gain = p_row / (1 + row @ p_row)
                w += torch.outer(gain, target - row @ w)
                p -= torch.outer(gain, p_row)
        with torch.no_grad():
            readout.weight.copy_(w[: readout.in_features].T)
            if readout.bias is not None:
                readout.bias.copy_(w[-1])
        return FitResult(rng=rng)
