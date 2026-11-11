"""An echo state network forecasting the Lorenz system.

A frozen leaky reservoir with a sparse recurrent matrix, read out by ridge
regression. Run with ``python -m examples.esn``.
"""

import torch
from torch import nn

from torchdyno import seed_all
from torchdyno.benchmark import lorenz_benchmark
from torchdyno.learners import Ridge, evaluate
from torchdyno.matrices import Dense, spectral, uniform
from torchdyno.sequence import LeakyCell, Recurrent

seed_all(0)
data = lorenz_benchmark(length=4000, target_delay=5, washout=100)

esn = nn.Sequential(
    Recurrent(
        LeakyCell(
            input=Dense(3, 500, init=uniform(1.0), bias=uniform(1.0)),
            recurrent=Dense(
                500, 500, init=spectral(lambda w: nn.init.sparse_(w, 0.9), 0.9)
            ),
        )
    ).requires_grad_(False),
    nn.Linear(500, 3),
)

l2 = torch.logspace(-8, -2, 4).tolist()
Ridge(l2, washout=data.washout, score_fn=data.task.primary).fit(
    esn, data.train_loader, data.val_loader
)
print(
    f"ESN test NRMSE: {evaluate(esn, data.eval_loader, data.task.primary, washout=data.washout):.2e}"
)
