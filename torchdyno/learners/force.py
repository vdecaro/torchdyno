"""FORCE: fit a fed-back readout online while the loop runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torchdyno.learners.features import find_readout
from torchdyno.learners.protocol import FitResult
from torchdyno.learners.rls import RLS
from torchdyno.reproducibility import get_rng_state
from torchdyno.sequence._checks import _absent
from torchdyno.sequence.cells.feedback import FeedbackCell

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import (
        Tensor,
        nn,
    )

    from torchdyno.matrices import Dense


class FORCE:
    """Fit the readout of a ``FeedbackCell`` in the loop.

    The model runs as it is, feeding back its own output, while recursive least
    squares updates the readout at every timestep on the new state and the
    target, so the fed-back output stays close to the target from the start
    (Sussillo and Abbott, 2009). ``P`` is kept across batches and fits, as by
    [`RLS`][torchdyno.learners.RLS], until [`reset()`][torchdyno.learners.FORCE.reset].
    """

    def __init__(
        self,
        l2: float = 1.0,
        washout: int = 0,
        cell: str | None = None,
        forgetting: float = 1.0,
    ):
        """Initializes the learner.

        Args:
            l2: the initial ``P = I / l2``, i.e. the ridge penalty.
            washout: leading timesteps run without an update.
            cell: the feedback cell's name in the model, as
                ``model.get_submodule`` takes it; ``None`` takes the only one.
            forgetting: the factor in ``(0, 1]`` older samples are discounted
                by at every update.

        Raises:
            ValueError: if ``l2`` is not positive or ``forgetting`` is outside
                ``(0, 1]``.
        """
        self.rls = RLS(l2=l2, forgetting=forgetting)
        self.washout = washout
        self.cell = cell

    def reset(self) -> None:
        """Forget ``P``: the next update starts from ``W = 0`` again."""
        self.rls.reset()

    def _find(self, model: nn.Module) -> FeedbackCell:
        if self.cell is not None:
            found = [model.get_submodule(self.cell)]
        else:
            found = [m for m in model.modules() if isinstance(m, FeedbackCell)]
        if len(found) != 1 or not isinstance(found[0], FeedbackCell):
            raise TypeError(
                f"Expected one FeedbackCell, found {len(found)}; name it with cell=."
            )
        return found[0]

    def _run(
        self,
        model: nn.Module,
        cell: FeedbackCell,
        readout: nn.Linear | Dense,
        x: Tensor,
        y: Tensor,
    ) -> int:
        """Run the model on one batch, updating at every step; the rows used."""
        present, step = ~_absent(x), iter(range(len(x)))

        def update(_cell: nn.Module, _args: object, h: Tensor) -> None:
            t = next(step)
            if t >= self.washout:
                self.rls.update(readout, h[present[t]], y[t][present[t]])

        handle = cell.register_forward_hook(update)
        try:
            with torch.no_grad():
                model(x)
        finally:
            handle.remove()
        return int(present[self.washout :].sum())

    def fit(
        self,
        model: nn.Module,
        train: Iterable[tuple[Tensor, Tensor]],
        val: Iterable[tuple[Tensor, Tensor]] | None = None,  # noqa: ARG002
    ) -> FitResult:
        """Run the model over ``train``, updating the readout at every step.

        Args:
            model: the model holding one feedback cell, run by a
                [`Recurrent`][torchdyno.sequence.Recurrent] layer.
            train: batches of time-first ``(x, y)``, ``y`` of shape ``(T, B, O)``
                with the readout's ``O`` outputs; ``x`` may have no features.
            val: unused; accepted so every learner fits the same way.

        Returns:
            The RNG state at the start.

        Raises:
            TypeError: if the model has no single feedback cell, or its readout
                is not a real ``nn.Linear`` or ``Dense``.
            ValueError: if a target does not match the input and the readout,
                or ``train`` has no samples.
        """
        cell = self._find(model)
        readout = find_readout(cell, "readout")
        rng = get_rng_state()
        seen = 0
        for x, y in train:
            if y.shape != (*x.shape[:2], readout.out_features):
                raise ValueError(
                    f"y must be (T, B, O) = {(*x.shape[:2], readout.out_features)}, "
                    f"got {tuple(y.shape)}."
                )
            seen += self._run(model, cell, readout, x, y)
        if not seen:
            raise ValueError("train has no samples to fit on.")
        return FitResult(rng=rng)
