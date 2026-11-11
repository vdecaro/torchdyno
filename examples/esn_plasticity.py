"""Intrinsic plasticity before the ridge readout of an echo state network.

A strong input drives about half of the reservoir's units into saturation. The
same reservoir is read out twice: as initialised, and after intrinsic
plasticity has adapted its gains and biases toward unsaturated outputs. Run
with ``python -m examples.esn_plasticity``.
"""

import copy

from torch import nn

from torchdyno import seed_all
from torchdyno.benchmark import lorenz_benchmark
from torchdyno.learners import IntrinsicPlasticity, Ridge, evaluate
from torchdyno.matrices import Dense, spectral, uniform
from torchdyno.sequence import LeakyCell, Recurrent

seed_all(0)
data = lorenz_benchmark(length=4000, target_delay=5, washout=100)
esn = nn.Sequential(
    Recurrent(
        LeakyCell(
            input=Dense(3, 300, init=uniform(6.0), bias=uniform(3.0)),
            recurrent=Dense(300, 300, init=spectral(uniform(1.0), 0.95)),
            leak=0.5,
        )
    ).requires_grad_(False),
    nn.Linear(300, 3),
)
adapted = copy.deepcopy(esn)
IntrinsicPlasticity(lr=1e-3, sigma=0.2).fit(adapted, data.train_loader)

for name, model in [("without", esn), ("with", adapted)]:
    Ridge(l2=1e-6, washout=data.washout).fit(model, data.train_loader)
    nrmse = evaluate(model, data.eval_loader, data.task.primary, washout=data.washout)
    print(f"{name} intrinsic plasticity: test NRMSE {nrmse:.2e}")
