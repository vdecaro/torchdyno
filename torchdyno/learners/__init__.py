"""Learners: ``fit(model, train, val=None) -> FitResult``, see :class:`Learner`."""

from .backprop import Backprop
from .chain import Chain
from .force import FORCE
from .plasticity import IntrinsicPlasticity
from .protocol import (
    FitResult,
    Learner,
)
from .quantile_bias import QuantileBias
from .ridge import Ridge
from .rls import RLS
from .scoring import evaluate

__all__ = [
    "FORCE",
    "RLS",
    "Backprop",
    "Chain",
    "FitResult",
    "IntrinsicPlasticity",
    "Learner",
    "QuantileBias",
    "Ridge",
    "evaluate",
]
