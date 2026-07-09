"""LRUCore: a Linear Recurrent Unit as a SequenceCore.

Orvieto et al. (2023), *Resurrecting Recurrent Neural Networks for Long
Sequences* (ICML). A complex diagonal linear recurrence with a stable
exponential parameterization and ring initialization, run by a parallel
associative scan. The complex→real readout ``C`` lives in the Head; the core
emits the real-ified state ``[Re(h); Im(h)] ∈ ℝ^{2N}``.
"""

import math
from typing import (
    Any,
    Optional,
    Tuple,
    Union,
)

import torch
import torch.nn.utils.parametrize as P
from torch import (
    Tensor,
    nn,
)

from torchdyno.models.base import (
    CoreCapabilities,
    CoreOutput,
    SequenceCore,
)
from torchdyno.nn.init import Ring
from torchdyno.nn.parametrize import StableExpComplex
from torchdyno.nn.scan import associative_scan
from torchdyno.registry import (
    ModelCard,
    register_core,
)

_INITS = {"ring": Ring}


def _resolve_init(init: Union[str, Ring]) -> Ring:
    if isinstance(init, str):
        if init not in _INITS:
            raise ValueError(f"Unknown init {init!r}. Available: {sorted(_INITS)}.")
        return _INITS[init]()
    return init


_LRU_CARD = ModelCard(
    name="lru",
    family="ssm",
    paper="Orvieto, A., Smith, S. L., Gu, A., Fernando, A., Gulcehre, C., "
    "Pascanu, R., & De, S. (2023). Resurrecting Recurrent Neural Networks for "
    "Long Sequences. ICML.",
    description="A Linear Recurrent Unit: a complex diagonal linear state-space "
    "recurrence with a stable exponential parameterization and ring "
    "initialization, run by a parallel associative scan. Frozen (ridge) or "
    "trained (backprop).",
    admits=("ridge", "backprop"),
    adapters=(),
    tasks=("forecast", "classify"),
    default_config={"input_size": 1, "n_modes": 64},
)


@register_core("lru", card=_LRU_CARD)
class LRUCore(SequenceCore):
    """A Linear Recurrent Unit as a :class:`SequenceCore`."""

    def __init__(
        self,
        input_size: int,
        n_modes: int = 64,
        *,
        trainable: bool = True,
        normalize: bool = True,
        init: Union[str, Ring] = "ring",
        generator: Optional[torch.Generator] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if input_size < 1:
            raise ValueError(f"input_size must be >= 1, got {input_size}.")
        if n_modes < 1:
            raise ValueError(f"n_modes must be >= 1, got {n_modes}.")

        nu0, theta0 = _resolve_init(init)(n_modes, generator=generator, dtype=dtype)

        # Register the stability parametrization, then set the surrogates exactly.
        self.lambda_ = nn.Parameter(torch.polar(torch.exp(-torch.exp(nu0)), theta0))
        P.register_parametrization(self, "lambda_", StableExpComplex())
        with torch.no_grad():
            self.parametrizations.lambda_.original0.copy_(nu0)
            self.parametrizations.lambda_.original1.copy_(theta0)

        std = 1.0 / math.sqrt(input_size)
        self.B_re = nn.Parameter(
            torch.randn(n_modes, input_size, generator=generator, dtype=dtype) * std
        )
        self.B_im = nn.Parameter(
            torch.randn(n_modes, input_size, generator=generator, dtype=dtype) * std
        )

        self.input_size = input_size
        self.n_modes = n_modes
        self.state_size = 2 * n_modes
        self.normalize = normalize

        if not trainable:
            for p in self.parameters():
                p.requires_grad_(False)

        self.capabilities = CoreCapabilities(
            compute_mode="scan",
            differentiable=trainable,
            trainable_recurrence=trainable,
            supports_step=True,
            admits_feedback=False,
            exposes_layer_states=False,
            dtype=dtype,
        )

    def _input_map(self) -> Tensor:
        """Effective (optionally γ-normalized) complex input matrix ``(N, input)``."""
        B = torch.complex(self.B_re, self.B_im)
        if self.normalize:
            gamma = torch.sqrt((1.0 - self.lambda_.abs() ** 2).clamp_min(0.0))
            B = gamma.unsqueeze(-1) * B
        return B

    def forward(
        self,
        x: Tensor,
        *,
        state0: Any = None,
        mask: Optional[Tensor] = None,
    ) -> CoreOutput:
        lam = self.lambda_  # (N,) complex
        B = self._input_map()  # (N, input)
        b = torch.einsum("tbi,ni->tbn", x.to(B.dtype), B)  # (T, B, N) complex
        if state0 is not None:
            b0 = b[:1] + lam.reshape(1, 1, -1) * state0.unsqueeze(0)
            b = torch.cat([b0, b[1:]], dim=0)
        a = lam.reshape(1, 1, -1).expand_as(b)  # time-invariant a_t = λ
        h = associative_scan(a, b)  # (T, B, N) complex
        states = torch.cat([h.real, h.imag], dim=-1)  # (T, B, 2N) real
        if mask is not None:
            states = mask * states
        return CoreOutput(states=states, final_state=h[-1])

    def step(self, x_t: Tensor, state: Any) -> Tuple[Tensor, Any]:
        lam = self.lambda_
        B = self._input_map()
        b_t = torch.einsum("bi,ni->bn", x_t.to(B.dtype), B)  # (B, N) complex
        h = b_t if state is None else lam * state + b_t
        emitted = torch.cat([h.real, h.imag], dim=-1)  # (B, 2N) real
        return emitted, h
