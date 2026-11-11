"""coRNN: coupled damped oscillators, trained end to end.

Run with ``python -m examples.cornn``.
"""

from torch import nn

from examples.data import frequency_task
from torchdyno import seed_all
from torchdyno.learners import Backprop, evaluate
from torchdyno.matrices import Dense, uniform
from torchdyno.objectives import Accuracy
from torchdyno.readout import Last
from torchdyno.sequence import OscillatorCell, Recurrent

seed_all(0)
train, _, test = frequency_task()
H = 64
cornn = nn.Sequential(
    Recurrent(
        OscillatorCell(
            input=Dense(1, H, bias=nn.init.zeros_),
            recurrent=Dense(2 * H, H, init=uniform(H**-0.5)),
            dt=0.05,
            gamma=2.7,
            epsilon=4.7,
        )
    ),
    Last(),
    nn.Linear(2 * H, 3),
)
Backprop(nn.CrossEntropyLoss(), lr=1e-2, epochs=30).fit(cornn, train)
print(f"coRNN test accuracy: {evaluate(cornn, test, Accuracy()):.4f}")
