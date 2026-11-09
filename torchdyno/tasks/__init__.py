"""Tasks: a loss, its metrics and the metric that selects."""

from .base import Task
from .classification import SequenceClassification
from .regression import Regression

__all__ = [
    "Regression",
    "SequenceClassification",
    "Task",
]
