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


class AdaDiagCore(SequenceCore):
    """A sparse assembly of contractive diagonal RNN modules (AdaDiag, eq. 9).

    Per-timestep update (⊙ = elementwise, φ = activation, xt = input[t]):
        Wφ = tanh(beta_W) ⊙ φ(v)                    # diagonal per-module weight
        ν  = lo + (hi-lo)·sigmoid(gate(xt) + α⊙v)   # per-unit characteristic freq
        v  ← v + ν ⊙ (−v + Wφ + L·v + B·xt)         # coupling + input inside ν

    Stability is guaranteed by construction: tanh bounds the diagonal weight to
    (−1, 1), the sigmoid bounds ν to (1e-5, 0.2), and the coupling is
    skew-symmetric, so no rescaling is needed.
    """

    def __init__(
        self,
        input_size: int,
        block_sizes: List[int],
        coupling_topology: Union[int, float, Literal["ring"], List[Tuple[int, int]]],
        coupling_block_init: Union[str, Callable[..., Tensor]] = "orthogonal",
        gate_bounds: Tuple[float, float] = (1e-5, 0.2),
        activation: str = "relu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if len(block_sizes) == 0:
            raise ValueError("At least one module (block) must be defined.")
        hidden = sum(block_sizes)
        self.input_size = input_size
        self.state_size = hidden
        self._dtype = dtype
        self.activ_fn = getattr(torch, activation)

        # Diagonal per-module weight (eq. 9): effective diagonal is tanh(beta_W).
        self.beta_W = nn.Parameter(
            torch.empty(hidden, dtype=dtype).uniform_(-1.0, 1.0)
        )
        self.couplings = build_coupling(
            block_sizes, coupling_topology, coupling_block_init, dtype
        )
        self.input_mat = nn.Parameter(
            torch.normal(
                mean=0.0,
                std=1.0 / math.sqrt(hidden),
                size=(hidden, input_size),
                dtype=dtype,
            )
        )
        self.gate = FrequencyGate(
            input_size, hidden, gate_bounds[0], gate_bounds[1], dtype
        )

        self.capabilities = CoreCapabilities(
            compute_mode="loop",
            differentiable=True,
            trainable_recurrence=True,
            supports_step=True,
            admits_feedback=False,
            exposes_layer_states=False,
            dtype=dtype,
        )

    def _update(self, x_t: Tensor, v: Tensor) -> Tensor:
        wphi = torch.tanh(self.beta_W) * self.activ_fn(v)
        lv = self.couplings(v)
        bu = F.linear(x_t, self.input_mat)
        nu = self.gate(x_t, v)
        return v + nu * (-v + wphi + lv + bu)

    def _initial_state(self, batch: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, self.state_size, dtype=self._dtype, device=device)

    def forward(
        self,
        x: Tensor,
        *,
        state0: Any = None,
        mask: Optional[Tensor] = None,
    ) -> CoreOutput:
        v = self._initial_state(x.shape[1], x.device) if state0 is None else state0
        states: List[Tensor] = []
        for t in range(x.shape[0]):
            v = self._update(x[t], v)
            states.append(v if mask is None else mask * v)
        return CoreOutput(states=torch.stack(states, dim=0), final_state=v)

    def step(self, x_t: Tensor, state: Any) -> Tuple[Tensor, Any]:
        v = self._initial_state(x_t.shape[0], x_t.device) if state is None else state
        v = self._update(x_t, v)
        return v, v
