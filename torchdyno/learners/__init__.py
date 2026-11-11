"""Learners: ``fit(model, train, val=None) -> FitResult``, see :class:`Learner`."""

from .backprop import Backprop
from .chain import Chain
from .plasticity import IntrinsicPlasticity
from .protocol import (
    FitResult,
    Learner,
)
from .ridge import Ridge
from .rls import RLS
from .scoring import evaluate

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
