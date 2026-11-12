"""Level 4, objectives: what a model is trained and scored for.

A [`Metric`][torchdyno.objectives.Metric] is a score with a selection direction; a
[`Task`][torchdyno.objectives.Task] bundles a loss, its metrics and the metric that
selects.
"""

from .base import (
    Metric,
    Task,
)
from .classification import (
    Accuracy,
    BalancedAccuracy,
    ConfusionMatrix,
    F1,
    SequenceClassification,
    TimestepClassification,
)
from .regression import (
    MSE,
    NRMSE,
    MemoryCapacity,
    Regression,
)

__all__ = [
    "F1",
    "MSE",
    "NRMSE",
    "Accuracy",
    "BalancedAccuracy",
    "ConfusionMatrix",
    "MemoryCapacity",
    "Metric",
    "Regression",
    "SequenceClassification",
    "Task",
    "TimestepClassification",
]
