"""Learners: ``fit(model, train, val=None) -> FitResult`` on a sequential model."""

from .backprop import Backprop
from .base import (
    FitResult,
    evaluate,
)

__all__ = [
    "Backprop",
    "FitResult",
    "evaluate",
]
