"""Task objects: a loss + metrics + a primary metric per problem shape."""

from .base import Task
from .regression import Regression
from .classification import SequenceClassification, DenseLabeling

__all__ = ["Task", "Regression", "SequenceClassification", "DenseLabeling"]
