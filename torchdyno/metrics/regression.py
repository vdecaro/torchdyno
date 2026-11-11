"""Regression metrics (lower is better)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torchdyno.metrics.base import (
    Metric,
    require_same_shape,
)

if TYPE_CHECKING:
    from torch import Tensor


class MSE(Metric):
    """Mean squared error."""

    def __call__(self, pred: Tensor, target: Tensor) -> float:
        """Score ``pred`` against ``target``.

        Args:
            pred: the predictions.
            target: the targets, same shape as ``pred``.

        Returns:
            The mean squared error.

        Raises:
            ValueError: if the shapes differ.
        """
        require_same_shape(pred, target)
        return torch.mean((pred - target) ** 2).item()


class NRMSE(Metric):
    """Normalized RMSE: ``sqrt(mse / (var(target) + eps))``."""

    def __init__(self, eps: float = 1e-8):
        """Initializes the metric.

        Args:
            eps: added to the variance to avoid dividing by zero.
        """
        self.eps = eps

    def __call__(self, pred: Tensor, target: Tensor) -> float:
        """Score ``pred`` against ``target``.

        Args:
            pred: the predictions.
            target: the targets, same shape as ``pred``.

        Returns:
            The RMSE divided by the standard deviation of the target.

        Raises:
            ValueError: if the shapes differ.
        """
        require_same_shape(pred, target)
        mse = torch.mean((pred - target) ** 2)
        var = torch.var(target, unbiased=False)
        return torch.sqrt(mse / (var + self.eps)).item()
