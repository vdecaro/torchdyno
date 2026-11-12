"""Cells: one step of a recurrence, ``cell(u_t, h) -> h'``.

The new state is also the cell's output, so a layer such as
[`Recurrent`][torchdyno.sequence.Recurrent] only has to collect the states. Cells take
their matrices as arguments, so the same update rule works with any matrix
structure from [`torchdyno.matrices`][torchdyno.matrices].
"""

from .base import Cell
from .closed_form import ClosedFormCell
from .euler import (
    EulerCell,
    Gate,
    LeakyField,
)
from .feedback import FeedbackCell
from .leaky import LeakyCell
from .linear import (
    LinearDiagonalCell,
    RingDiagonal,
    ZOHDiagonal,
)
from .oscillator import OscillatorCell

__all__ = [
    "Cell",
    "ClosedFormCell",
    "EulerCell",
    "FeedbackCell",
    "Gate",
    "LeakyCell",
    "LeakyField",
    "LinearDiagonalCell",
    "OscillatorCell",
    "RingDiagonal",
    "ZOHDiagonal",
]
