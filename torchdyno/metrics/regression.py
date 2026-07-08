"""Regression metrics (lower is better)."""

import torch
from torch import Tensor

from torchdyno.metrics.base import Metric


class MSE(Metric):
    """Mean squared error."""

    name = "mse"
    mode = "min"

    def __call__(self, pred: Tensor, target: Tensor) -> float:
        return torch.mean((pred - target) ** 2).item()


class MAE(Metric):
    """Mean absolute error."""

    name = "mae"
    mode = "min"

    def __call__(self, pred: Tensor, target: Tensor) -> float:
        return torch.mean(torch.abs(pred - target)).item()


class NRMSE(Metric):
    """Normalized RMSE: ``sqrt(mse / (var(target) + eps))``."""

    name = "nrmse"
    mode = "min"

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, pred: Tensor, target: Tensor) -> float:
        mse = torch.mean((pred - target) ** 2)
        var = torch.var(target, unbiased=False)
        return torch.sqrt(mse / (var + self.eps)).item()
