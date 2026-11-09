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
