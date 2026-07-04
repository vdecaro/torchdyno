"""AdaDiag: a sparse assembly of contractive-by-design diagonal RNN modules.

Reference: Ceni, De Caro, Bacciu, Gallicchio, "Sparse Assemblies of Recurrent
Neural Networks with Stability Guarantees" (github.com/andreaceni/AdaDiag).
"""

import math
from typing import (
    Any,
    Callable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import torch
import torch.nn.functional as F
from torch import (
    Tensor,
    nn,
)

from torchdyno.models.assembly._common import build_coupling
from torchdyno.models.base import (
    CoreCapabilities,
    CoreOutput,
    SequenceCore,
)


class FrequencyGate(nn.Module):
    """Per-unit characteristic frequencies ν, bounded to ``(lo, hi)``.

    ``ν = lo + (hi - lo) * sigmoid(gate_input(x_t) + gate_state * v)`` where
    ``gate_input`` is a dense input projection with bias (Γ^τ, β^τ) and
    ``gate_state`` is a per-unit diagonal state coupling (α^τ) initialized to
    zeros so ν starts input-only.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        lo: float = 1e-5,
        hi: float = 0.2,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.lo = lo
        self.hi = hi
        self.gate_input = nn.Linear(input_size, hidden_size, dtype=dtype)
        self.gate_state = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, x_t: Tensor, v: Tensor) -> Tensor:
        logits = self.gate_input(x_t) + self.gate_state * v
        return self.lo + (self.hi - self.lo) * torch.sigmoid(logits)
