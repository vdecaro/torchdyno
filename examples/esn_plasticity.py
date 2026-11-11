"""Intrinsic plasticity before the ridge readout of an echo state network.

The same reservoir is read out twice: as initialised, and after intrinsic
plasticity has adapted its gains and biases. Run with
``python -m examples.esn_plasticity``.
"""

import copy

from torch import nn

from torchdyno import seed_all
from torchdyno.benchmark import lorenz_benchmark
from torchdyno.cells import LeakyCell
from torchdyno.layers import Recurrent
from torchdyno.learners import IntrinsicPlasticity, Ridge, evaluate
from torchdyno.matrices import Dense, spectral, uniform

seed_all(0)
data = lorenz_benchmark(length=4000, target_delay=5, washout=100)
esn = nn.Sequential(
    Recurrent(
        LeakyCell(
            input=Dense(3, 300, init=uniform(1.0), bias=uniform(0.5)),
            recurrent=Dense(300, 300, init=spectral(uniform(1.0), 0.95)),
            leak=0.5,
        )
    ).requires_grad_(False),
    nn.Linear(300, 3),
)
adapted = copy.deepcopy(esn)
IntrinsicPlasticity(lr=0.5, mu=0.0, sigma=0.2, epochs=10).fit(
    adapted, data.train_loader
)

for name, model in [("without", esn), ("with", adapted)]:
    Ridge(l2=1e-6, washout=data.washout).fit(model, data.train_loader)
    nrmse = evaluate(model, data.eval_loader, data.task.primary, washout=data.washout)
    print(f"{name} intrinsic plasticity: test NRMSE {nrmse:.2e}")
