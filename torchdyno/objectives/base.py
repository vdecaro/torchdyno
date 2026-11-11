"""What a metric and a task are."""

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
    from collections.abc import Callable

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


class Task:
    """Bundles a task's loss and metrics.

    A trainer uses ``loss``, a solver selecting on validation uses ``primary``
    (a :class:`Metric`, hence a score with a ``mode``), and reports use every
    metric in ``metrics``. Build one directly, e.g. ``Task({"f1": F1(),
    "accuracy": Accuracy()}, "f1", loss=F.cross_entropy)``, or subclass it and
    override :meth:`loss`.
    """

    def __init__(
        self,
        metrics: dict[str, Metric],
        primary: str,
        loss: Callable[[Tensor, Tensor], Tensor] | None = None,
    ):
        """Initializes the task.

        Args:
            metrics: the metrics to report, by name.
            primary: the name of the metric that selects on validation.
            loss: ``loss(pred, target) -> Tensor``, the training objective.

        Raises:
            ValueError: if ``primary`` is not in ``metrics``.
        """
        if primary not in metrics:
            raise ValueError(
                f"Unknown primary metric {primary!r}. Choose from {sorted(metrics)}."
            )
        self._metrics = metrics
        self.primary = metrics[primary]
        self._loss = loss

    def loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """The differentiable training objective.

        Args:
            pred: the predictions.
            target: the targets.

        Returns:
            The scalar loss.

        Raises:
            NotImplementedError: if the task was built without ``loss``.
        """
        if self._loss is None:
            raise NotImplementedError("Pass loss= to the task, or override loss().")
        return self._loss(pred, target)

    def metrics(self, pred: Tensor, target: Tensor) -> dict[str, float]:
        """Every metric of the task.

        Args:
            pred: the predictions.
            target: the targets.

        Returns:
            The score of each metric, by name.
        """
        return {name: metric(pred, target) for name, metric in self._metrics.items()}
