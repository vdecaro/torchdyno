"""Top-level package for TorchDyno.

A model is a plain ``nn.Sequential`` of parts from four levels:
:mod:`~torchdyno.matrices`, :mod:`~torchdyno.sequence`,
:mod:`~torchdyno.readout` and :mod:`~torchdyno.objectives`, fitted by
:mod:`~torchdyno.learners`.
"""

from . import (
    benchmark,
    learners,
    matrices,
    objectives,
    readout,
    sequence,
    testing,
)
from .reproducibility import (
    get_rng_state,
    seed_all,
    set_rng_state,
)

__all__ = [
    "benchmark",
    "get_rng_state",
    "learners",
    "matrices",
    "objectives",
    "readout",
    "seed_all",
    "sequence",
    "set_rng_state",
    "testing",
]

__version__ = "0.2.3"
