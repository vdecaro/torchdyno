"""What every learner returns, and the scoring they share."""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

import torch

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
    )

    from torch import (
        Tensor,
        nn,
    )

Score = TypeVar("Score")


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


def evaluate(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    score_fn: Callable[[Tensor, Tensor], Score],
) -> Score:
    """Score a model over a whole loader at once.

    Args:
        model: the model to score.
        loader: batches of time-first ``(x, y)``.
        score_fn: ``score_fn(prediction, target)``, returning any score.

    Returns:
        The score of the predictions for every batch, concatenated.
    """
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x))
            targets.append(y)
    # time-first (T, B, O) predictions batch on dim 1; pooled (B, C) on dim 0
    dim = 1 if preds[0].ndim == 3 else 0  # noqa: PLR2004
    return score_fn(torch.cat(preds, dim=dim), torch.cat(targets, dim=dim))
