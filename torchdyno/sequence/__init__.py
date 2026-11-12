"""Level 2, sequences: cells, and layers from (T, B, F) to (T, B, H) sequences.

A timestep whose features are all NaN is one a sequence does not have, e.g. the
padding of a shorter sequence in a batch, at its end or its start. The layers
skip such steps and output NaN at them, and pooling and the learners skip them
in turn, so sequences of different lengths share a batch.
"""

from .cells import (
    Cell,
    ClosedFormCell,
    EulerCell,
    FeedbackCell,
    Gate,
    LeakyCell,
    LeakyDiagonal,
    LeakyField,
    LinearDiagonalCell,
    OscillatorCell,
    RingDiagonal,
    ZOHDiagonal,
)
from .combine import (
    Concat,
    Stack,
    Sum,
)
from .convolution import (
    Kernels,
    RandomConv,
    minirocket,
    rocket,
)
from .recurrent import (
    Recurrent,
    Scan,
)
from .time import (
    Diff,
    Reverse,
)

__all__ = [
    "Cell",
    "ClosedFormCell",
    "Concat",
    "Diff",
    "EulerCell",
    "FeedbackCell",
    "Gate",
    "Kernels",
    "LeakyCell",
    "LeakyDiagonal",
    "LeakyField",
    "LinearDiagonalCell",
    "OscillatorCell",
    "RandomConv",
    "Recurrent",
    "Reverse",
    "RingDiagonal",
    "Scan",
    "Stack",
    "Sum",
    "ZOHDiagonal",
    "minirocket",
    "rocket",
]
