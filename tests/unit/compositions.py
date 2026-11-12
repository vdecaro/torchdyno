"""Known models composed from parts, at test sizes."""

import torch
from torch import nn

from torchdyno.matrices import (
    BlockDiagonal,
    Dense,
    Diagonal,
    SkewCoupling,
    normal,
    polar,
    spectral,
    uniform,
)
from torchdyno.sequence import (
    ClosedFormCell,
    EulerCell,
    Gate,
    LeakyCell,
    LeakyDiagonal,
    LeakyField,
    LinearDiagonalCell,
    OscillatorCell,
    Recurrent,
    RingDiagonal,
    ZOHDiagonal,
)


def adadiag(features=2, blocks=(4, 4, 4), pairs=2, dtype=torch.float32):
    size = sum(blocks)
    return Recurrent(
        EulerCell(
            LeakyField(
                recurrent=Diagonal(size, bound=torch.tanh, dtype=dtype),
                coupling=SkewCoupling(list(blocks), pairs=pairs, dtype=dtype),
                input=Dense(features, size, init=normal(size**-0.5), dtype=dtype),
                activation=torch.relu,
            ),
            step=Gate(features, size, dtype=dtype),
        )
    )


def scn(features=2, blocks=(4, 4, 4), pairs="ring", step=0.1, dtype=torch.float32):
    size = sum(blocks)
    return Recurrent(
        EulerCell(
            LeakyField(
                recurrent=BlockDiagonal(list(blocks), bound=torch.tanh, dtype=dtype),
                coupling=SkewCoupling(list(blocks), pairs=pairs, dtype=dtype),
                input=Dense(features, size, init=normal(size**-0.5), dtype=dtype),
            ),
            step=step,
        )
    )


def reservoir(features=2, size=16, leak=0.5, rho=0.9, dtype=torch.float32):
    return Recurrent(
        LeakyCell(
            input=Dense(
                features, size, init=uniform(0.9), bias=uniform(0.9), dtype=dtype
            ),
            recurrent=Dense(size, size, init=spectral(uniform(1.0), rho), dtype=dtype),
            leak=leak,
        )
    )


def lru(features=2, modes=8, dtype=torch.float32):
    return Recurrent(
        LinearDiagonalCell(
            input=Dense(
                features, modes, init=normal(features**-0.5), dtype=dtype.to_complex()
            ),
            transition=RingDiagonal(modes, r_min=0.4, r_max=0.99, dtype=dtype),
        )
    )


def paralesn(features=2, modes=8, leak=0.5, dtype=torch.float32):
    complex_dtype = dtype.to_complex()
    return Recurrent(
        LinearDiagonalCell(
            input=Dense(features, modes, init=uniform(1.0), dtype=complex_dtype),
            transition=LeakyDiagonal(
                Diagonal(modes, init=polar((0.5, 0.9)), dtype=complex_dtype), leak
            ),
        )
    )


def s4d(features=2, modes=8, dtype=torch.float32):
    return Recurrent(
        LinearDiagonalCell(
            input=Dense(
                features, modes, init=normal(features**-0.5), dtype=dtype.to_complex()
            ),
            transition=ZOHDiagonal(modes, dtype=dtype),
        )
    )


def cornn(features=2, size=8, dtype=torch.float32):
    return Recurrent(
        OscillatorCell(
            input=Dense(features, size, bias=nn.init.zeros_, dtype=dtype),
            recurrent=Dense(2 * size, size, init=uniform(size**-0.5), dtype=dtype),
            dt=0.1,
            gamma=1.0,
            epsilon=1.0,
        )
    )


def cfc(features=2, size=8, units=8, dtype=torch.float32):
    return Recurrent(
        ClosedFormCell(
            backbone=nn.Sequential(
                nn.Linear(size + features, units, dtype=dtype), nn.Tanh()
            ),
            heads=nn.Linear(units, 4 * size, dtype=dtype),
            dt=1.0,
        )
    )


COMPOSITIONS = {
    "adadiag": adadiag,
    "scn": scn,
    "reservoir": reservoir,
    "lru": lru,
    "paralesn": paralesn,
    "s4d": s4d,
    "cornn": cornn,
    "cfc": cfc,
}
