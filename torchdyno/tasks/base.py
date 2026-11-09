"""The Task base: a loss, a metric set and the metric that selects."""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

    from torchdyno.metrics import Metric


class Task(ABC):
    """Bundles a task's loss and metrics.

    A trainer uses ``loss``, a solver selecting on validation uses ``primary``
    (a :class:`Metric`, hence a score with a ``mode``), and reports use every
    metric in ``metrics``.
    """

    def __init__(self, metrics: dict[str, Metric], primary: str):
        """Initializes the task.

        Args:
            metrics: the metrics to report, by name.
            primary: the name of the metric that selects on validation.

        Raises:
            ValueError: if ``primary`` is not in ``metrics``.
        """
        if primary not in metrics:
            raise ValueError(
                f"Unknown primary metric {primary!r}. Choose from {sorted(metrics)}."
            )
        self._metrics = metrics
        self.primary = metrics[primary]

    @abstractmethod
    def loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """The differentiable training objective.

        Args:
            pred: the predictions.
            target: the targets.

        Returns:
            The scalar loss.
        """
        raise NotImplementedError

    def metrics(self, pred: Tensor, target: Tensor) -> dict[str, float]:
        """Every metric of the task.

        Args:
            pred: the predictions.
            target: the targets.

        Returns:
            The score of each metric, by name.
        """
        return {name: metric(pred, target) for name, metric in self._metrics.items()}
