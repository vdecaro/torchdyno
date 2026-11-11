"""What every learner implements and returns."""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
    )

    from torch import (
        Tensor,
        nn,
    )


@dataclass
class FitResult:
    """The outcome of fitting a model.

    Args:
        history: per-epoch metrics (e.g. ``{"train_loss": [...]}``); empty for
            one-shot solvers.
        best: selected hyperparameters and scores (e.g. ``{"l2": 1e-6}``).
        rng: RNG state captured at fit start, for
            :func:`~torchdyno.reproducibility.set_rng_state`.
    """

    history: dict[str, list[float]] = field(default_factory=dict)
    best: dict[str, Any] = field(default_factory=dict)
    rng: dict[str, Any] | None = None


class Learner(Protocol):
    """Anything that fits a model: what every learner here implements.

    A new learning rule needs only this method to run wherever a learner is
    expected, e.g. in :class:`Chain` or :func:`~torchdyno.benchmark.run`.
    """

    def fit(
        self,
        model: nn.Module,
        train: Iterable[tuple[Tensor, Tensor]],
        val: Iterable[tuple[Tensor, Tensor]] | None = None,
    ) -> FitResult:
        """Fit ``model`` in place.

        Args:
            model: the model to fit.
            train: batches of time-first ``(x, y)``.
            val: optional batches for selection or monitoring.

        Returns:
            What the fit recorded.
        """
        ...
