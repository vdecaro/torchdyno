"""Level 4, objectives: what a model is trained and scored for.

A :class:`Metric` is a score with a selection direction; a :class:`Task` bundles a
loss, its metrics and the metric that selects.
"""

from .base import (
    Metric,
    Task,
)
from .classification import (
    Accuracy,
    SequenceClassification,
)
from .regression import (
    MSE,
    NRMSE,
    Regression,
)

__all__ = [
    "MSE",
    "NRMSE",
    "Accuracy",
    "Metric",
    "Regression",
    "SequenceClassification",
    "Task",
]
