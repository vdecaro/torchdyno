"""Package for optimization functionalities."""

from . import ridge_regression as rr
from .adapters import CoreAdapter
from .intrinsic_plasticity import IntrinsicPlasticity
from .ridge import RidgeRegression

__all__ = ["rr", "CoreAdapter", "IntrinsicPlasticity", "RidgeRegression"]
