"""AdaDiag: a sparse assembly of diagonal modules with an adaptive step, trained end to end.

Each unit has a bounded diagonal self-weight, the modules are coupled by a
skew-symmetric matrix, and a gate sets every unit's Euler step from the input
and the state. Run with ``python -m examples.adadiag``.
"""

import torch
from torch import nn

from examples.data import frequency_task
from torchdyno import seed_all
from torchdyno.cells import EulerCell, Gate, LeakyField
from torchdyno.layers import Recurrent
from torchdyno.learners import Backprop, evaluate
from torchdyno.matrices import Dense, Diagonal, SkewCoupling, normal
from torchdyno.metrics import Accuracy
from torchdyno.pooling import Last

seed_all(0)
train, _, test = frequency_task()
H = 16 * 8
adadiag = nn.Sequential(
    Recurrent(
        EulerCell(
            LeakyField(
                recurrent=Diagonal(H, bound=torch.tanh),
                coupling=SkewCoupling([16] * 8, pairs=8),
                input=Dense(1, H, init=normal(H**-0.5)),
                activation=torch.relu,
            ),
            step=Gate(1, H, bounds=(1e-5, 0.2)),
        )
    ),
    Last(),
    nn.Linear(H, 3),
)
Backprop(nn.CrossEntropyLoss(), lr=1e-2, epochs=30).fit(adadiag, train)
print(f"AdaDiag test accuracy: {evaluate(adadiag, test, Accuracy()):.4f}")
