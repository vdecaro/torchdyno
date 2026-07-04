"""Prediction heads that map core states to task outputs."""

from .base import Head
from .regression import RegressionHead

__all__ = ["Head", "RegressionHead"]
