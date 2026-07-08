"""Classification tasks: cross-entropy loss + classification metrics."""

from typing import Dict

import torch.nn.functional as F
from torch import Tensor

from torchdyno.metrics import Accuracy, MacroF1, Metric, to_indices
from torchdyno.tasks.base import Task


class _ClassificationTask(Task):
    """Shared metric set + primary selection for the classification tasks."""

    def __init__(self, num_classes: int, primary: str = "accuracy"):
        self.num_classes = num_classes
        self._metrics = {"accuracy": Accuracy(), "macro_f1": MacroF1()}
        self._primary = self._select_primary(self._metrics, primary)

    def metrics(self, pred: Tensor, target: Tensor) -> Dict[str, float]:
        return {name: metric(pred, target) for name, metric in self._metrics.items()}

    @property
    def primary(self) -> Metric:
        return self._primary


class SequenceClassification(_ClassificationTask):
    """One label per sequence. Pairs with :class:`ClassificationHead` ``(B, C)``."""

    def loss(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(pred, to_indices(pred, target))


class DenseLabeling(_ClassificationTask):
    """One label per timestep. Pairs with :class:`DenseHead` ``(T, B, C)``."""

    def loss(self, pred: Tensor, target: Tensor) -> Tensor:
        num_classes = pred.shape[-1]
        idx = to_indices(pred, target)
        return F.cross_entropy(pred.reshape(-1, num_classes), idx.reshape(-1))
