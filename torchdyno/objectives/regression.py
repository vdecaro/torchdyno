"""Regression: squared errors, and time-aligned targets."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
)

import torch

from torchdyno.objectives.base import (
    Metric,
    Task,
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
    """Normalized RMSE: ``sqrt(mse / (var(target) + eps))``, averaged over outputs.

    The last dimension holds the outputs, each normalised by its own variance;
    a 1-D target is a single output.
    """

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
            Each output's RMSE divided by its standard deviation, averaged.

        Raises:
            ValueError: if the shapes differ.
        """
        require_same_shape(pred, target)
        if target.ndim == 1:
            pred, target = pred[:, None], target[:, None]
        pred, target = pred.flatten(0, -2), target.flatten(0, -2)
        mse = torch.mean((pred - target) ** 2, dim=0)
        var = torch.var(target, dim=0, unbiased=False)
        return torch.sqrt(mse / (var + self.eps)).mean().item()


class MemoryCapacity(Metric):
    """Jaeger's memory capacity: ``Σ_k r²(ŷ_k, y_k)`` over the outputs.

    Output ``k`` recalls the input ``k`` steps back, as in
    :func:`~torchdyno.benchmark.memory_capacity_benchmark`; each adds its
    squared correlation with its target, so ``K`` outputs score at most ``K``.
    """

    mode = "max"

    def __call__(self, pred: Tensor, target: Tensor) -> float:
        """Score ``pred`` against ``target``.

        Args:
            pred: the predictions, one output per delay in the last dimension.
            target: the delayed inputs, same shape as ``pred``.

        Returns:
            The sum of the squared correlations; a constant output adds zero.

        Raises:
            ValueError: if the shapes differ.
        """
        require_same_shape(pred, target)
        if target.ndim == 1:
            pred, target = pred[:, None], target[:, None]
        pred, target = pred.flatten(0, -2).double(), target.flatten(0, -2).double()
        pred, target = pred - pred.mean(0), target - target.mean(0)
        covariance = (pred * target).sum(0)
        r2 = covariance**2 / ((pred**2).sum(0) * (target**2).sum(0))
        return r2.nan_to_num(0.0).sum().item()


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
