"""CfCCore: a Closed-form Continuous-time (CfC) cell as a SequenceCore.

Hasani et al. (2022). Closed-form continuous-time neural networks. Nature Machine
Intelligence. A gated closed-form approximation of Liquid Time-Constant networks: a
learned, time-modulated gate blends two tanh candidate states over an optional shared
MLP backbone. Emits x (size N); states bounded in (-1, 1) by construction. Frozen
(bounded reservoir) or trained (backprop).
"""

import math
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
from torchdyno.registry import (
    ModelCard,
    register_core,
)

_ACT = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}

_CFC_CARD = ModelCard(
    name="cfc",
    family="ode",
    paper="Hasani, R., Lechner, M., Amini, A., Liebenwein, L., Ray, A., Tschaikowski, "
    "M., Teschl, G., & Rus, D. (2022). Closed-form continuous-time neural networks. "
    "Nature Machine Intelligence.",
    description="A Closed-form Continuous-time (CfC) cell: a learned time-modulated gate "
    "blends two tanh candidate states, x_new = ff1*(1-gate) + ff2*gate with "
    "gate = sigmoid(t_a*dt + t_b). Frozen (bounded reservoir -> ridge) or trained "
    "(backprop).",
    admits=("ridge", "backprop"),
    adapters=(),
    tasks=("forecast", "classify"),
    default_config={"input_size": 1, "hidden_size": 64},
)


@register_core("cfc", card=_CFC_CARD)
class CfCCore(SequenceCore):
    """A Closed-form Continuous-time (CfC) cell as a :class:`SequenceCore`."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        *,
        trainable: bool = True,
        backbone_units: Optional[int] = None,
        backbone_layers: int = 1,
        activation: str = "tanh",
        dt: float = 1.0,
        generator: Optional[torch.Generator] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if input_size < 1:
            raise ValueError(f"input_size must be >= 1, got {input_size}.")
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {hidden_size}.")
        if backbone_layers < 0:
            raise ValueError(f"backbone_layers must be >= 0, got {backbone_layers}.")
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}.")
        if activation not in _ACT:
            raise ValueError(
                f"Unknown activation {activation!r}. Choose from {sorted(_ACT)}."
            )

        units = backbone_units if backbone_units is not None else hidden_size
        if backbone_layers > 0 and units < 1:
            raise ValueError(f"backbone_units must be >= 1, got {units}.")

        cat_dim = input_size + hidden_size
        if backbone_layers == 0:
            self.backbone = nn.Identity()
            head_in = cat_dim
        else:
            act = _ACT[activation]
            layers = [nn.Linear(cat_dim, units, dtype=dtype), act()]
            for _ in range(backbone_layers - 1):
                layers += [nn.Linear(units, units, dtype=dtype), act()]
            self.backbone = nn.Sequential(*layers)
            head_in = units

        self.head_ff1 = nn.Linear(head_in, hidden_size, dtype=dtype)
        self.head_ff2 = nn.Linear(head_in, hidden_size, dtype=dtype)
        self.head_ta = nn.Linear(head_in, hidden_size, dtype=dtype)
        self.head_tb = nn.Linear(head_in, hidden_size, dtype=dtype)

        # Seed all Linear params from the generator (matches nn.Linear's default init
        # bound = 1/sqrt(fan_in)) so the core is seed-reproducible.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                bound = 1.0 / math.sqrt(m.weight.shape[1])
                with torch.no_grad():
                    m.weight.uniform_(-bound, bound, generator=generator)
                    if m.bias is not None:
                        m.bias.uniform_(-bound, bound, generator=generator)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.state_size = hidden_size
        self._dtype = dtype
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

    def _update(self, u_t: Tensor, x: Tensor) -> Tensor:
        b = self.backbone(torch.cat([x, u_t], dim=-1))
        ff1 = torch.tanh(self.head_ff1(b))
        ff2 = torch.tanh(self.head_ff2(b))
        gate = torch.sigmoid(self.head_ta(b) * self.dt + self.head_tb(b))
        return ff1 * (1.0 - gate) + ff2 * gate

    def _initial_state(self, batch: int, device: torch.device) -> Tensor:
        return torch.zeros(batch, self.hidden_size, dtype=self._dtype, device=device)

    def forward(
        self,
        x: Tensor,
        *,
        state0: Any = None,
        mask: Optional[Tensor] = None,
    ) -> CoreOutput:
        h = self._initial_state(x.shape[1], x.device) if state0 is None else state0
        states = []
        for t in range(x.shape[0]):
            h = self._update(x[t], h)
            states.append(h)
        out = torch.stack(states, dim=0)  # (T, B, N)
        if mask is not None:
            out = mask * out
        return CoreOutput(states=out, final_state=h)

    def step(self, x_t: Tensor, state: Any) -> Tuple[Tensor, Any]:
        h = self._initial_state(x_t.shape[0], x_t.device) if state is None else state
        h = self._update(x_t, h)
        return h, h
