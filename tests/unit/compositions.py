"""Known models composed from parts, at test sizes."""

import torch

from torchdyno.cells import (
    EulerCell,
    Gate,
    LeakyCell,
)
from torchdyno.layers import Recurrent
from torchdyno.matrices import (
    BlockDiagonal,
    Dense,
    Diagonal,
    SkewCoupling,
    normal,
    spectral,
    uniform,
)


def adadiag(features=2, blocks=(4, 4, 4), pairs=2, dtype=torch.float32):
    size = sum(blocks)
    return Recurrent(
        EulerCell(
            recurrent=Diagonal(size, bound=torch.tanh, dtype=dtype),
            coupling=SkewCoupling(list(blocks), pairs=pairs, dtype=dtype),
            input=Dense(features, size, init=normal(size**-0.5), dtype=dtype),
            step=Gate(features, size, dtype=dtype),
            activation=torch.relu,
        )
    )


def scn(features=2, blocks=(4, 4, 4), pairs="ring", step=0.1, dtype=torch.float32):
    size = sum(blocks)
    return Recurrent(
        EulerCell(
            recurrent=BlockDiagonal(list(blocks), bound=torch.tanh, dtype=dtype),
            coupling=SkewCoupling(list(blocks), pairs=pairs, dtype=dtype),
            input=Dense(features, size, init=normal(size**-0.5), dtype=dtype),
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


COMPOSITIONS = {"adadiag": adadiag, "scn": scn, "reservoir": reservoir}
