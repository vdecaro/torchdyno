"""S4D as a frozen feature extractor, pooled like ROCKET and read out by ridge.

The modes are linear in the input, so the nonlinear PPV and max pooling of
ROCKET's features is what separates the frequencies. The same composition
trains end to end with ``Backprop``; here only the readout is fitted. Run with
``python -m examples.s4d``.
"""

import torch
from torch import nn

from examples.data import frequency_task
from torchdyno import seed_all
from torchdyno.learners import Ridge, evaluate
from torchdyno.matrices import Dense, normal
from torchdyno.objectives import Accuracy
from torchdyno.readout import PPV, Max
from torchdyno.sequence import Concat, LinearDiagonalCell, Scan, ZOHDiagonal

seed_all(0)
train, val, test = frequency_task()
N = 128
s4d = nn.Sequential(
    Scan(
        LinearDiagonalCell(
            transition=ZOHDiagonal(N, dt_min=1e-2, dt_max=1.0),
            input=Dense(1, N, init=normal(1.0), dtype=torch.cfloat),
        )
    ).requires_grad_(False),
    Concat(PPV(), Max()),
    nn.Linear(4 * N, 3),
)
l2 = torch.logspace(-4, 2, 7).tolist()
Ridge(l2, standardize=True, score_fn=Accuracy()).fit(s4d, train, val)
print(f"S4D test accuracy: {evaluate(s4d, test, Accuracy()):.4f}")
