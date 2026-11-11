"""Learners: ``fit(model, train, val=None) -> FitResult``, see :class:`Learner`."""

from .backprop import Backprop
from .plasticity import IntrinsicPlasticity
from .base import (
    Chain,
    FitResult,
    Learner,
    evaluate,
)
from .ridge import Ridge
from .rls import RLS

__all__ = [
    "RLS",
    "Backprop",
    "Chain",
    "FitResult",
    "IntrinsicPlasticity",
    "Learner",
    "Ridge",
    "evaluate",
]
