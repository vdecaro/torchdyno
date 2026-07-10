"""DiagonalSSMCore: shared machinery for diagonal state-space cores.

A diagonal SSM runs a stable diagonal linear recurrence ``h_t = a ⊙ h_{t-1} + B̄·u_t`` via a
parallel associative scan, emitting the real-ified state ``[Re(h); Im(h)] ∈ ℝ^{2N}``.
Subclasses supply only the discrete diagonal coefficient and effective input map via
``_recurrence``; the base owns forward/step/emission, the shared input matrix ``B``, the
trainable/frozen wiring, and the capability declaration.
"""

import math
from abc import abstractmethod
from typing import (
    Any,
    Optional,
    Tuple,
)

import torch
from torch import (
    Tensor,
    nn,
)

from torchdyno.models.base import (
    CoreCapabilities,
    CoreOutput,
    SequenceCore,
)
from torchdyno.nn.scan import associative_scan


class DiagonalSSMCore(SequenceCore):
    """Base class for diagonal state-space cores (LRU, S4D, ...)."""

    def __init__(
        self,
        input_size: int,
        n_modes: int,
        *,
        trainable: bool,
        dtype: torch.dtype,
    ):
        super().__init__()
        if input_size < 1:
            raise ValueError(f"input_size must be >= 1, got {input_size}.")
        if n_modes < 1:
            raise ValueError(f"n_modes must be >= 1, got {n_modes}.")
        self.input_size = input_size
        self.n_modes = n_modes
        self.state_size = 2 * n_modes
        self._trainable = trainable
        self.capabilities = CoreCapabilities(
            compute_mode="scan",
            differentiable=trainable,
            trainable_recurrence=trainable,
            supports_step=True,
            admits_feedback=False,
            exposes_layer_states=False,
            dtype=dtype,
        )

    def _make_B(self, generator: Optional[torch.Generator], dtype: torch.dtype) -> None:
        """Create the shared complex input matrix ``B_re``/``B_im`` ~ ``N(0, 1/input_size)``.

        The subclass calls this in ``__init__`` at the point that preserves its intended
        RNG-draw order (kept out of the base ``__init__`` so reparenting a core does not
        reorder its random initialization).
        """
        std = 1.0 / math.sqrt(self.input_size)
        self.B_re = nn.Parameter(
            torch.randn(self.n_modes, self.input_size, generator=generator, dtype=dtype)
            * std
        )
        self.B_im = nn.Parameter(
            torch.randn(self.n_modes, self.input_size, generator=generator, dtype=dtype)
            * std
        )

    def _finalize(self) -> None:
        """Freeze all parameters when the core is not trainable. Call LAST in ``__init__``."""
        if not self._trainable:
            for p in self.parameters():
                p.requires_grad_(False)

    @abstractmethod
    def _recurrence(self) -> Tuple[Tensor, Tensor]:
        """Return ``(a_diag: (N,) complex, B_eff: (N, input_size) complex)``."""
        raise NotImplementedError

    def forward(
        self,
        x: Tensor,
        *,
        state0: Any = None,
        mask: Optional[Tensor] = None,
    ) -> CoreOutput:
        a_diag, B = self._recurrence()
        b = torch.einsum("tbi,ni->tbn", x.to(B.dtype), B)
        if state0 is not None:
            b0 = b[:1] + a_diag.reshape(1, 1, -1) * state0.unsqueeze(0)
            b = torch.cat([b0, b[1:]], dim=0)
        a = a_diag.reshape(1, 1, -1).expand_as(b)
        h = associative_scan(a, b)
        states = torch.cat([h.real, h.imag], dim=-1)
        if mask is not None:
            states = mask * states
        return CoreOutput(states=states, final_state=h[-1])

    def step(self, x_t: Tensor, state: Any) -> Tuple[Tensor, Any]:
        a_diag, B = self._recurrence()
        b_t = torch.einsum("bi,ni->bn", x_t.to(B.dtype), B)
        h = b_t if state is None else a_diag * state + b_t
        emitted = torch.cat([h.real, h.imag], dim=-1)
        return emitted, h
