"""SCN: fixed block-diagonal modules coupled by a trained skew-symmetric matrix.

The blocks are frozen with ``requires_grad_(False)``; the coupling, the input
and the readout train. Run with ``python -m examples.scn``.
"""

import torch
from torch import nn

from examples.data import frequency_task
from torchdyno import seed_all
from torchdyno.learners import Backprop, evaluate
from torchdyno.matrices import BlockDiagonal, Dense, SkewCoupling, normal
from torchdyno.objectives import Accuracy
from torchdyno.readout import Last
from torchdyno.sequence import EulerCell, LeakyField, Recurrent

seed_all(0)
train, _, test = frequency_task()
H = 16 * 8
scn = nn.Sequential(
    Recurrent(
        EulerCell(
            LeakyField(
                recurrent=BlockDiagonal([16] * 8, bound=torch.tanh),
                coupling=SkewCoupling([16] * 8, pairs="ring"),
                input=Dense(1, H, init=normal(H**-0.5)),
            ),
            step=0.1,
        )
    ),
    Last(),
    nn.Linear(H, 3),
)
scn[0].cell.field.recurrent.requires_grad_(False)
Backprop(nn.CrossEntropyLoss(), lr=1e-2, epochs=30).fit(scn, train)
print(f"SCN test accuracy: {evaluate(scn, test, Accuracy()):.4f}")
