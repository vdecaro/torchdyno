"""RLS: fit a linear readout online, by recursive least squares."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torchdyno.learners.features import (
    feature_rows,
    find_readout,
)
from torchdyno.learners.protocol import FitResult
from torchdyno.reproducibility import get_rng_state

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import (
        Tensor,
        nn,
    )

    from torchdyno.matrices import Dense


class RLS:
    """Fit a linear readout on whatever reaches it, one sample at a time.

    The readout is chosen as for :class:`~torchdyno.learners.Ridge`. Rows are
    visited in order: every timestep of every sequence for time-aligned
    features, one row per sequence for pooled ones. The first update starts
    from ``W = 0`` and ``P = I / l2``, so one pass with ``forgetting=1`` gives
    the ridge solution with penalty ``l2``. Later fits and updates continue from
    the readout's weights and the kept ``P``, until :meth:`reset`. A bias is
    fitted as the weight of a constant feature, so unlike
    :class:`~torchdyno.learners.Ridge` it is penalised too. Each update costs
    ``O(H²)``.
    """

    def __init__(
        self,
        l2: float = 1.0,
        washout: int = 0,
        readout: str | None = None,
        forgetting: float = 1.0,
    ):
        """Initializes the learner.

        Args:
            l2: the initial ``P = I / l2``, i.e. the ridge penalty.
            washout: leading timesteps to discard; only for time-aligned
                features.
            readout: the readout's name in the model, as
                ``model.get_submodule`` takes it; ``None`` takes ``model[-1]``.
            forgetting: the factor in ``(0, 1]`` older samples are discounted
                by at every update; ``1`` forgets nothing.

        Raises:
            ValueError: if ``l2`` is not positive or ``forgetting`` is outside
                ``(0, 1]``.
        """
        if not l2 > 0:
            raise ValueError(f"l2 must be > 0, got {l2}.")
        if not 0 < forgetting <= 1:
            raise ValueError(f"forgetting must be in (0, 1], got {forgetting}.")
        self.l2 = l2
        self.washout = washout
        self.readout = readout
        self.forgetting = forgetting
        self.p: Tensor | None = None

    def reset(self) -> None:
        """Forget ``P``: the next update starts from ``W = 0`` again."""
        self.p = None

    def fit(
        self,
        model: nn.Module,
        train: Iterable[tuple[Tensor, Tensor]],
        val: Iterable[tuple[Tensor, Tensor]] | None = None,  # noqa: ARG002
    ) -> FitResult:
        """Update the readout over every row of ``train``.

        Args:
            model: the model whose readout is set in place.
            train: batches of time-first ``(x, y)``, as for
                :class:`~torchdyno.learners.Ridge`.
            val: unused; accepted so every learner fits the same way.

        Returns:
            The RNG state at the start.

        Raises:
            TypeError: if the readout is not a real ``nn.Linear`` or ``Dense``.
            ValueError: if ``train`` has no samples.
        """
        readout = find_readout(model, self.readout)
        rng = get_rng_state()
        seen = 0
        for x, y in train:
            rows, targets = feature_rows(model, readout, x, y, self.washout)
            self.update(readout, rows, targets)
            seen += len(rows)
        if not seen:
            raise ValueError("train has no samples to fit on.")
        return FitResult(rng=rng)

    @torch.no_grad()
    def update(self, readout: nn.Linear | Dense, rows: Tensor, targets: Tensor) -> None:
        """Update ``readout`` in place on rows of features and targets, in order.

        The step a FORCE loop calls at every timestep, with the readout's input
        and the target of its output.

        Args:
            readout: the readout to update.
            rows: ``(n, H)`` features.
            targets: ``(n, O)`` targets.

        Raises:
            ValueError: if the kept ``P`` is for a readout of another size.
        """
        bias = readout.bias is not None
        size = readout.in_features + bias
        f64, device = torch.float64, readout.weight.device
        if self.p is None:
            self.p = torch.eye(size, dtype=f64, device=device) / self.l2
            w = torch.zeros(size, readout.out_features, dtype=f64, device=device)
        else:
            if self.p.shape[0] != size:
                raise ValueError(
                    f"P is for a readout of {self.p.shape[0]} inputs, not {size}; "
                    "reset() first."
                )
            w = readout.weight.detach().T.to(torch.float64, copy=True)
            if readout.bias is not None:
                w = torch.cat([w, readout.bias.detach().to(torch.float64)[None]])
        rows, targets = rows.to(torch.float64), targets.to(torch.float64)
        if bias:
            rows = torch.cat([rows, rows.new_ones(len(rows), 1)], dim=1)
        p = self.p
        for row, target in zip(rows, targets, strict=True):
            p_row = p @ row
            gain = p_row / (self.forgetting + row @ p_row)
            w += torch.outer(gain, target - row @ w)
            p = (p - torch.outer(gain, p_row)) / self.forgetting
        self.p = p
        readout.weight.copy_(w[: readout.in_features].T)
        if readout.bias is not None:
            readout.bias.copy_(w[-1])
