"""SCNCore: a block-diagonal contractive assembly (Sparse Combo Net baseline).

Salvages the legacy ``RNNAssembly`` dynamics onto the SequenceCore contract:
a fixed Euler step over a block-diagonal recurrence coupled by a skew-symmetric
matrix. With ``constrained_blocks="fixed"`` the block weights are frozen and
only the couplings (and the external readout) train — the original SCN of
Kozachkov et al. (2022), the baseline AdaDiag is compared against.
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

from torchdyno.models import initializers
from torchdyno.models.assembly._common import build_coupling
from torchdyno.models.base import (
    CoreCapabilities,
    CoreOutput,
    SequenceCore,
)
from torchdyno.models.rnn_assembly import BlockDiagonal


class SCNCore(SequenceCore):
    """A block-diagonal contractive assembly integrated with a fixed Euler step."""

    def __init__(
        self,
        input_size: int,
        block_sizes: List[int],
        coupling_topology: Union[int, float, Literal["ring"], List[Tuple[int, int]]],
        block_init: Union[str, Callable[..., Tensor]] = "orthogonal",
        coupling_block_init: Union[str, Callable[..., Tensor]] = "orthogonal",
        eul_step: float = 1e-2,
        activation: str = "relu",
        constrained_blocks: Optional[
            Literal["fixed", "tanh", "clip", "orthogonal"]
        ] = "fixed",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if len(block_sizes) == 0:
            raise ValueError("At least one module (block) must be defined.")
        hidden = sum(block_sizes)
        self.input_size = input_size
        self.state_size = hidden
        self._dtype = dtype
        self._eul_step = eul_step
        self.activ_fn = getattr(torch, activation)

        if isinstance(block_init, str):
            block_init_fn: Callable[..., Tensor] = getattr(initializers, block_init)
        else:
            block_init_fn = block_init
        blocks = [block_init_fn((b, b), dtype=dtype) for b in block_sizes]
        self.blocks = BlockDiagonal(blocks=blocks, constrained=constrained_blocks)

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
        return v + self._eul_step * (
            -v
            + self.blocks(self.activ_fn(v))
            + self.couplings(v)
            + F.linear(x_t, self.input_mat)
        )

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
