"""The Metric base: a scalar score with a selection direction."""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    TYPE_CHECKING,
    Literal,
)

if TYPE_CHECKING:
    from torch import Tensor


class Metric(ABC):
    """A scalar score, callable as ``metric(pred, target) -> float``.

    ``mode`` says whether a smaller (``"min"``) or larger (``"max"``) value is
    better, so a learner can select a hyperparameter on a validation set.
    """

    mode: Literal["min", "max"] = "min"

    @abstractmethod
    def __call__(self, pred: Tensor, target: Tensor) -> float:
        """Score ``pred`` against ``target``.

        Args:
            pred: the predictions.
            target: the targets.

        Returns:
            The score.
        """
        raise NotImplementedError


def require_same_shape(pred: Tensor, target: Tensor) -> None:
    """Raise unless ``pred`` and ``target`` have the same shape.

    Broadcasting a ``(B, 1)`` prediction against a ``(B,)`` target would score
    every prediction against every target.

    Args:
        pred: the predictions.
        target: the targets.

    Raises:
        ValueError: if the shapes differ.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred {tuple(pred.shape)} and target {tuple(target.shape)} must have "
            "the same shape."
        )
