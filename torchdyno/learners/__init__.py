"""Learners: ``fit(model, train, val=None) -> FitResult`` on a sequential model."""

from .backprop import Backprop
from .base import (
    FitResult,
    evaluate,
)
from .ridge import Ridge

__all__ = [
    "Backprop",
    "FitResult",
    "Ridge",
    "evaluate",
]
