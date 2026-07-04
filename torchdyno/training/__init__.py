"""Loop-owning trainers and the shared Learner surface."""

from .backprop import BackpropTrainer
from .base import FitResult, Learner

__all__ = ["BackpropTrainer", "FitResult", "Learner"]
