"""Chain: run learners one after the other."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

from torchdyno.learners.protocol import FitResult

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
    )

    from torch import (
        Tensor,
        nn,
    )

    from torchdyno.learners.protocol import Learner


class Chain:
    """Run learners one after the other on the same model.

    E.g. ``Chain(IntrinsicPlasticity(), Ridge())`` adapts a reservoir, then
    fits its readout; ``Chain(Ridge(), Backprop(...))`` fine-tunes from the
    closed-form solution.
    """

    def __init__(self, *learners: Learner):
        """Initializes the chain.

        Args:
            learners: the learners, in order.
        """
        self.learners = learners

    def fit(
        self,
        model: nn.Module,
        train: Iterable[tuple[Tensor, Tensor]],
        val: Iterable[tuple[Tensor, Tensor]] | None = None,
    ) -> FitResult:
        """Fit ``model`` with every learner in turn.

        Args:
            model: the model to fit in place.
            train: batches of time-first ``(x, y)``, given to every learner.
            val: optional batches, given to every learner.

        Returns:
            Every learner's history and selections, keyed ``"<index>.<name>"``,
            and the RNG state at the start.
        """
        chained = FitResult()
        for i, learner in enumerate(self.learners):
            result = learner.fit(model, train, val)
            chained.history |= {f"{i}.{k}": v for k, v in result.history.items()}
            chained.best |= {f"{i}.{k}": v for k, v in result.best.items()}
            chained.rng = chained.rng or result.rng
        return chained
