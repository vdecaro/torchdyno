"""Regression task: MSE loss, NRMSE and MSE metrics."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
)

import torch

from torchdyno.metrics import (
    MSE,
    NRMSE,
)
from torchdyno.metrics.base import require_same_shape
from torchdyno.tasks.base import Task

if TYPE_CHECKING:
    from torch import Tensor


class Regression(Task):
    """Point regression on time-aligned ``(T, B, O)`` targets."""

    def __init__(self, primary: Literal["nrmse", "mse"] = "nrmse"):
        """Initializes the task.

        Args:
            primary: the metric that selects on validation.
        """
        super().__init__({"nrmse": NRMSE(), "mse": MSE()}, primary)

    def loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Mean squared error.

        Args:
            pred: the predictions.
            target: the targets, same shape as ``pred``.

        Returns:
            The scalar loss.

        Raises:
            ValueError: if the shapes differ.
        """
        require_same_shape(pred, target)
        return torch.mean((pred - target) ** 2)
