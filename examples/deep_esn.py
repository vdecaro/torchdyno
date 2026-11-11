"""A deep echo state network on Lorenz, read out over every layer's states.

``Stack`` runs the reservoirs in series and concatenates their outputs; a plain
``nn.Sequential`` of the same reservoirs would read only the last one. Run with
``python -m examples.deep_esn``.
"""

from torch import nn

from torchdyno import seed_all
from torchdyno.benchmark import lorenz_benchmark
from torchdyno.cells import LeakyCell
from torchdyno.layers import Recurrent, Stack
from torchdyno.learners import Ridge, evaluate
from torchdyno.matrices import Dense, spectral, uniform


def reservoir(inputs, size):
    return Recurrent(
        LeakyCell(
            input=Dense(inputs, size, init=uniform(0.5)),
            recurrent=Dense(size, size, init=spectral(uniform(1.0), 0.9)),
            leak=0.5,
        )
    )


seed_all(0)
data = lorenz_benchmark(length=4000, target_delay=5)
deep = nn.Sequential(
    Stack(reservoir(3, 200), reservoir(200, 200), reservoir(200, 200)).requires_grad_(
        False
    ),
    nn.Linear(600, 3),
)
Ridge(l2=1e-6, washout=100).fit(deep, data.train_loader)
print(
    f"deep ESN test NRMSE: {evaluate(deep, data.eval_loader, data.task.primary, washout=100):.2e}"
)
