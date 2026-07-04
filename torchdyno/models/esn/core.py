"""ESNCore: a stack of frozen reservoirs exposed as a SequenceCore.

Mirrors the reservoir-stacking behavior of the legacy ``EchoStateNetwork``
(each layer is chained into the next; ``"stacked"`` reads out the last layer,
``"multi"``/``"parallel"`` concatenate all layers) but exposes it through the
:class:`SequenceCore` contract instead of owning a readout or training.
"""

from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
from torch import (
    Size,
    Tensor,
    nn,
)

from torchdyno.models.base import (
    CoreCapabilities,
    CoreOutput,
    SequenceCore,
)
from torchdyno.models.esn.reservoir import Reservoir

_ARCH_TYPES = ("stacked", "multi", "parallel")


class ESNCore(SequenceCore):
    """A chain of Echo State Network reservoirs as a :class:`SequenceCore`."""

    def __init__(
        self,
        input_size: int,
        layer_sizes: List[int],
        arch_type: str = "stacked",
        activation: str = "tanh",
        leakage: float = 1.0,
        input_scaling: float = 0.9,
        rho: float = 0.99,
        bias: bool = False,
        kernel_initializer: Union[str, Callable[[Size], Tensor]] = "uniform",
        recurrent_initializer: Union[str, Callable[[Size], Tensor]] = "uniform",
        net_gain_and_bias: bool = False,
    ):
        super().__init__()
        if layer_sizes is None or len(layer_sizes) == 0:
            raise ValueError("At least one hidden layer must be defined.")
        if arch_type not in _ARCH_TYPES:
            raise ValueError(
                f"Unknown architecture type: {arch_type}. "
                f"Choose from {_ARCH_TYPES}."
            )
        if len(layer_sizes) > 1 and net_gain_and_bias:
            raise ValueError(
                "Net gain and bias can only be used with one hidden layer."
            )

        self.arch_type = arch_type
        self.reservoirs = nn.ModuleList(
            [
                Reservoir(
                    input_size if i == 0 else layer_sizes[i - 1],
                    layer_sizes[i],
                    activation,
                    leakage,
                    input_scaling,
                    rho,
                    bias,
                    kernel_initializer,
                    recurrent_initializer,
                    net_gain_and_bias,
                )
                for i in range(len(layer_sizes))
            ]
        )

        self.input_size = input_size
        if arch_type == "stacked":
            self.state_size = layer_sizes[-1]
        else:
            self.state_size = sum(layer_sizes)
        self.capabilities = CoreCapabilities(
            compute_mode="loop",
            differentiable=False,
            trainable_recurrence=False,
            supports_step=True,
            admits_feedback=False,
            exposes_layer_states=True,
            dtype=torch.float32,
        )

    def forward(
        self,
        x: Tensor,
        *,
        state0: Any = None,
        mask: Optional[Tensor] = None,
    ) -> CoreOutput:
        layer_states: List[Tensor] = []
        signal = x
        for i, reservoir in enumerate(self.reservoirs):
            init = None if state0 is None else state0[i]
            signal = reservoir(signal, init, mask)
            layer_states.append(signal)

        if self.arch_type in ("multi", "parallel"):
            states = torch.cat(layer_states, dim=-1)
        else:
            states = layer_states[-1]

        final_state = [layer[-1] for layer in layer_states]
        return CoreOutput(
            states=states, final_state=final_state, layer_states=layer_states
        )

    def step(self, x_t: Tensor, state: Any) -> Tuple[Tensor, Any]:
        if state is None:
            state = [None] * len(self.reservoirs)
        new_state: List[Tensor] = []
        outs: List[Tensor] = []
        signal = x_t
        for i, reservoir in enumerate(self.reservoirs):
            # Reuse the reservoir's own recurrence on a length-1 sequence.
            seq = reservoir(signal.unsqueeze(0), state[i])
            s = seq[0]
            new_state.append(s)
            outs.append(s)
            signal = s

        if self.arch_type in ("multi", "parallel"):
            emitted = torch.cat(outs, dim=-1)
        else:
            emitted = outs[-1]
        return emitted, new_state
