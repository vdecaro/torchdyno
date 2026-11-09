"""CfC: a closed-form continuous-time cell with a small backbone, trained end to end.

Run with ``python -m examples.cfc``.
"""

from torch import nn

from examples.data import frequency_task
from torchdyno import seed_all
from torchdyno.cells import ClosedFormCell
from torchdyno.layers import Recurrent
from torchdyno.learners import Backprop, evaluate
from torchdyno.metrics import Accuracy
from torchdyno.pooling import Last

seed_all(0)
train, _, test = frequency_task()
H, units = 32, 128
cfc = nn.Sequential(
    Recurrent(
        ClosedFormCell(
            backbone=nn.Sequential(nn.Linear(H + 1, units), nn.Tanh()),
            heads=nn.Linear(units, 4 * H),
            dt=1.0,
        )
    ),
    Last(),
    nn.Linear(H, 3),
)
Backprop(nn.CrossEntropyLoss(), lr=3e-3, epochs=100).fit(cfc, train)
print(f"CfC test accuracy: {evaluate(cfc, test, Accuracy()):.4f}")
