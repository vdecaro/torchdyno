"""ParalESN on Lorenz: deep reservoir layers of linear complex modes, run by a parallel scan.

Each layer is a frozen leaky diagonal recurrence, then a convolution across its
modes and a tanh, its only nonlinearity. From Pinna et al., "ParalESN: Enabling
parallel information processing in Reservoir Computing" (2026). Run with
``python -m examples.paralesn``.
"""

import torch
from torch import nn

from torchdyno import seed_all
from torchdyno.benchmark import lorenz_benchmark
from torchdyno.learners import Ridge, evaluate
from torchdyno.matrices import Dense, Diagonal, polar, uniform
from torchdyno.sequence import LeakyDiagonal, LinearDiagonalCell, Mix, Scan, Stack


def layer(inputs, size, leak=0.5):
    eigenvalues = Diagonal(size, init=polar((0.5, 0.9)), dtype=torch.cfloat)
    # each mode reads its input scaled by sqrt(1 - |λ|²)
    scale = torch.sqrt(1 - eigenvalues.diagonal().detach().abs() ** 2)[:, None]
    return nn.Sequential(
        Scan(
            LinearDiagonalCell(
                input=Dense(
                    inputs,
                    size,
                    init=uniform(scale),
                    bias=uniform(0.1),
                    dtype=torch.cfloat,
                ),
                transition=LeakyDiagonal(eigenvalues, leak),
            )
        ),
        Mix(channels=2, kernel_size=3),
        nn.Tanh(),
    )


seed_all(0)
data = lorenz_benchmark(length=4000, target_delay=5, washout=100)
paralesn = nn.Sequential(
    Stack(layer(3, 200), layer(200, 200), layer(200, 200)).requires_grad_(False),
    nn.Linear(600, 3),
)
Ridge(l2=1e-6, washout=data.washout).fit(paralesn, data.train_loader)
print(
    f"ParalESN test NRMSE: {evaluate(paralesn, data.eval_loader, data.task.primary, washout=data.washout):.2e}"
)
