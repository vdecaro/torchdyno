"""Regression task: MSE loss + regression metrics."""

from typing import Dict

import torch
from torch import Tensor

from torchdyno.metrics import MAE, MSE, NRMSE, Metric
from torchdyno.tasks.base import Task


class Regression(Task):
    """Point regression. Loss = MSE; metrics = NRMSE/MSE/MAE.

    Args:
        primary: which metric selects on validation — ``"nrmse"`` (default),
            ``"mse"``, or ``"mae"``.
    """

    def __init__(self, primary: str = "nrmse"):
        self._metrics = {"nrmse": NRMSE(), "mse": MSE(), "mae": MAE()}
        if primary not in self._metrics:
            raise ValueError(
                f"Unknown primary metric {primary!r}. Choose from {sorted(self._metrics)}."
            )
        self._primary = self._metrics[primary]

    def loss(self, pred: Tensor, target: Tensor) -> Tensor:
        return torch.mean((pred - target) ** 2)

    def metrics(self, pred: Tensor, target: Tensor) -> Dict[str, float]:
        return {name: metric(pred, target) for name, metric in self._metrics.items()}

    @property
    def primary(self) -> Metric:
        return self._primary
