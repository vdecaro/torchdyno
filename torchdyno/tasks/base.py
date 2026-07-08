"""The Task base: problem semantics — a loss, a metric set, a primary metric."""

from abc import ABC, abstractmethod
from typing import Dict

from torch import Tensor

from torchdyno.metrics import Metric


class Task(ABC):
    """Bundles a task's loss and metrics.

    A trainer pulls ``loss`` (a differentiable objective); a solver/validator
    pulls ``primary`` (a :class:`Metric`, hence a ``score_fn`` carrying a
    ``mode``). ``metrics`` reports every metric for logging.
    """

    @abstractmethod
    def loss(self, pred: Tensor, target: Tensor) -> Tensor: ...

    @abstractmethod
    def metrics(self, pred: Tensor, target: Tensor) -> Dict[str, float]: ...

    @property
    @abstractmethod
    def primary(self) -> Metric: ...

    def _select_primary(self, metrics: Dict[str, "Metric"], primary: str) -> "Metric":
        """Validate ``primary`` against ``metrics`` and return the chosen Metric."""
        if primary not in metrics:
            raise ValueError(
                f"Unknown primary metric {primary!r}. Choose from {sorted(metrics)}."
            )
        return metrics[primary]
