"""Level 1, matrices: how a weight matrix is structured, initialised and applied.

Every matrix is a module whose ``forward(x)`` applies ``x Wᵀ`` without building
``W`` when its structure allows it, and whose ``dense()`` returns ``W``. A
matrix's ``weight`` is its storage, which can differ from ``W``: masked,
bounded, or, for a complex matrix, the real and imaginary parts in a last
dimension of 2, so that casts such as ``.double()`` reach both parts.
``torch.nn.Linear`` works wherever a matrix is expected. Initialisers follow the
``torch.nn.init`` convention, callables that fill a tensor in place, so every
``torch.nn.init`` function can be passed directly. Freeze a matrix with
``requires_grad_(False)``; freezing never changes its initialisation.
"""

from .blocks import (
    BlockDiagonal,
    SkewCoupling,
)
from .dense import Dense
from .diagonal import Diagonal
from .init import (
    normal,
    spectral,
    uniform,
)
from .ops import (
    LinearMap,
    to_dense,
    update_,
)

__all__ = [
    "BlockDiagonal",
    "Dense",
    "Diagonal",
    "LinearMap",
    "SkewCoupling",
    "normal",
    "spectral",
    "to_dense",
    "uniform",
    "update_",
]
