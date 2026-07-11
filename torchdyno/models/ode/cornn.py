"""coRNNCore: a Coupled Oscillatory RNN as a SequenceCore.

Rusch, T. K., & Mishra, S. (2021). Coupled Oscillatory Recurrent Neural Network
(coRNN): An accurate and (gradient) stable architecture for learning long time
dependencies. ICLR. A network of coupled, damped, controlled oscillators — a
second-order ODE discretized by symplectic (IMEX) Euler over the augmented state
(y = position, z = velocity). Emits the full state [y; z] ∈ ℝ^{2N}; gradient-stable
by construction. Frozen (oscillatory reservoir) or trained (backprop).
"""

import math
from typing import (
    Any,
    Optional,
    Tuple,
)

import torch
import torch.nn.functional as F
from torch import (
    Tensor,
    nn,
)

from torchdyno.models.base import (
    CoreCapabilities,
    CoreOutput,
    SequenceCore,
)
from torchdyno.registry import (
    ModelCard,
    register_core,
)

_CORNN_CARD = ModelCard(
    name="cornn",
    family="ode",
    paper="Rusch, T. K., & Mishra, S. (2021). Coupled Oscillatory Recurrent Neural "
    "Network (coRNN): An accurate and (gradient) stable architecture for learning "
    "long time dependencies. ICLR.",
    description="A network of coupled, damped, controlled oscillators (a second-order "
    "ODE discretized by symplectic Euler); gradient-stable by construction. Frozen "
    "(oscillatory reservoir -> ridge) or trained (backprop).",
    admits=("ridge", "backprop"),
    adapters=(),
    tasks=("forecast", "classify"),
    default_config={"input_size": 1, "hidden_size": 32},
)


@register_core("cornn", card=_CORNN_CARD)
class coRNNCore(SequenceCore):
    """A Coupled Oscillatory RNN as a :class:`SequenceCore`."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        *,
        trainable: bool = True,
        gamma: float = 1.0,
        epsilon: float = 1.0,
        dt: float = 0.1,
        activation: str = "tanh",
        generator: Optional[torch.Generator] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if input_size < 1:
            raise ValueError(f"input_size must be >= 1, got {input_size}.")
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}.")
        if gamma <= 0.0:
            raise ValueError(f"gamma must be > 0, got {gamma}.")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}.")
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}.")

        self.activ_fn = getattr(torch, activation)

        bound_h = 1.0 / math.sqrt(hidden_size)
        bound_i = 1.0 / math.sqrt(input_size)
        self.W = nn.Parameter(
            torch.empty(hidden_size, hidden_size, dtype=dtype).uniform_(
                -bound_h, bound_h, generator=generator
            )
        )
        self.W_z = nn.Parameter(
            torch.empty(hidden_size, hidden_size, dtype=dtype).uniform_(
                -bound_h, bound_h, generator=generator
            )
        )
        self.V = nn.Parameter(
            torch.empty(hidden_size, input_size, dtype=dtype).uniform_(
                -bound_i, bound_i, generator=generator
            )
        )
        self.b = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.state_size = 2 * hidden_size
        self._dtype = dtype
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.dt = float(dt)

        if not trainable:
            for p in self.parameters():
                p.requires_grad_(False)

        self.capabilities = CoreCapabilities(
            compute_mode="loop",
            differentiable=trainable,
            trainable_recurrence=trainable,
            supports_step=True,
            admits_feedback=False,
            exposes_layer_states=False,
            dtype=dtype,
        )

    def _update(self, x_t: Tensor, y: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:
        pre = self.activ_fn(
            F.linear(y, self.W) + F.linear(z, self.W_z) + F.linear(x_t, self.V) + self.b
        )
        z_new = z + self.dt * (pre - self.gamma * y - self.epsilon * z)
        y_new = y + self.dt * z_new
        return y_new, z_new

    def _initial_state(self, batch: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        zeros = torch.zeros(batch, self.hidden_size, dtype=self._dtype, device=device)
        return zeros, zeros.clone()

    def forward(
        self,
        x: Tensor,
        *,
        state0: Any = None,
        mask: Optional[Tensor] = None,
    ) -> CoreOutput:
        y, z = self._initial_state(x.shape[1], x.device) if state0 is None else state0
        states = []
        for t in range(x.shape[0]):
            y, z = self._update(x[t], y, z)
            states.append(torch.cat([y, z], dim=-1))  # (B, 2N)
        out = torch.stack(states, dim=0)  # (T, B, 2N)
        if mask is not None:
            out = mask * out
        return CoreOutput(states=out, final_state=(y, z))

    def step(self, x_t: Tensor, state: Any) -> Tuple[Tensor, Any]:
        y, z = self._initial_state(x_t.shape[0], x_t.device) if state is None else state
        y, z = self._update(x_t, y, z)
        return torch.cat([y, z], dim=-1), (y, z)
