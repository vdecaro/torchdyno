"""LRU: a linear complex diagonal recurrence run with a parallel scan, trained end to end.

The recurrence is linear, so a nonlinearity after it, here a position-wise
``nn.Linear`` and ``nn.GELU`` as in the LRU paper, is what lets the model tell
frequencies apart. Run with ``python -m examples.lru``.
"""

import math

import torch
from torch import nn

from examples.data import frequency_task
from torchdyno import seed_all
from torchdyno.learners import Backprop, evaluate
from torchdyno.matrices import Dense, normal
from torchdyno.objectives import Accuracy
from torchdyno.readout import Mean
from torchdyno.sequence import LinearDiagonalCell, RingDiagonal, Scan

seed_all(0)
train, _, test = frequency_task()
N = 64
lru = nn.Sequential(
    Scan(
        LinearDiagonalCell(
            input=Dense(1, N, init=normal(1.0), dtype=torch.cfloat),
            transition=RingDiagonal(N, r_min=0.4, r_max=0.99, max_phase=math.pi),
        )
    ),
    nn.Linear(2 * N, 64),
    nn.GELU(),
    Mean(),
    nn.Linear(64, 3),
)
Backprop(nn.CrossEntropyLoss(), lr=1e-2, epochs=30).fit(lru, train)
print(f"LRU test accuracy: {evaluate(lru, test, Accuracy()):.4f}")
