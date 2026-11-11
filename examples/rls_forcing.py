"""An online readout trained with teacher forcing, then run in closed loop.

The reservoir reads the true previous Lorenz state while recursive least
squares fits the readout online. In closed loop the prediction is fed back as
the next input, one step at a time, continuing the reservoir from its last
state. Run with ``python -m examples.rls_forcing``.
"""

import torch
from torch import nn

from torchdyno import seed_all
from torchdyno.benchmark import lorenz_system
from torchdyno.learners import RLS
from torchdyno.matrices import Dense, spectral, uniform
from torchdyno.sequence import LeakyCell, Recurrent

seed_all(0)
series = lorenz_system(3000).unsqueeze(1)  # (T, 1, 3)
teacher, target = series[:-1], series[1:]
train = 2000

reservoir = Recurrent(
    LeakyCell(
        input=Dense(3, 300, init=uniform(0.5)),
        recurrent=Dense(300, 300, init=spectral(uniform(1.0), 0.9)),
        leak=0.5,
    )
).requires_grad_(False)
model = nn.Sequential(reservoir, nn.Linear(300, 3))
RLS(l2=1e-3, washout=100).fit(model, [(teacher[:train], target[:train])])

with torch.no_grad():
    state = reservoir(teacher[:train])[-1]
    prediction, generated = target[train - 1], []
    for _ in range(100):
        state = reservoir(prediction[None], h0=state)[-1]
        prediction = model[-1](state)
        generated.append(prediction)
    error = (
        (torch.stack(generated) - target[train : train + 100]).norm(dim=-1).squeeze(-1)
    )
print(
    f"closed-loop error after 10, 50, 100 steps: {error[9]:.4f}, {error[49]:.4f}, {error[99]:.4f}"
)
