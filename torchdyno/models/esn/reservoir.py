import logging
from typing import (
    Callable,
    List,
    Optional,
    Union,
)

import torch
import torch.nn.functional as F
from torch import (
    Generator,
    Size,
    Tensor,
)
from torch.nn import (
    Module,
    Parameter,
)

from torchdyno.models import initializers


class Reservoir(Module):
    """A Reservoir of for Echo State Networks.

    Args:
        input_size: the number of expected features in the input `x`
        hidden_size: the number of features in the hidden state `h`
        activation: name of the activation function from `torch` (e.g. `torch.tanh`)
        leakage: the value of the leaking parameter `alpha`
        input_scaling: the value for the desired scaling of the input (must be `<= 1`)
        rho: the desired spectral radius of the recurrent matrix (must be `< 1`)
        bias: if ``False``, the layer does not use bias weights `b`
        mode: execution mode of the reservoir (vanilla or intrinsic plasticity)
        kernel_initializer: the kind of initialization of the input transformation.
            Default: `'uniform'`
        recurrent_initializer: the kind of initialization of the recurrent matrix.
            Default: `'normal'`
        net_gain_and_bias: if ``True``, the network uses additional ``g`` (gain)
            and ``b`` (bias) parameters. Default: ``False``
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
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
        if input_scaling > 1:
            logging.warning("Input scaling must be <= 1")
        if rho >= 1:
            logging.warning("Spectral radius must be < 1")

        self.input_scaling = Parameter(torch.tensor(input_scaling), requires_grad=False)
        self.rho = Parameter(torch.tensor(rho), requires_grad=False)

        if isinstance(kernel_initializer, str):
            kernel_initializer_ = getattr(initializers, kernel_initializer)
        else:
            kernel_initializer_ = kernel_initializer

        if isinstance(recurrent_initializer, str):
            recurrent_initializer_ = getattr(initializers, recurrent_initializer)
        else:
            recurrent_initializer_ = recurrent_initializer

        self.W_in = Parameter(
            initializers.rescale(
                kernel_initializer_([hidden_size, input_size]), "linear", input_scaling
            ),
            requires_grad=False,
        )
        self.W_hat = Parameter(
            initializers.rescale(
                recurrent_initializer_([hidden_size, hidden_size]), "spectral", rho
            ),
            requires_grad=False,
        )
        self.b = (
            Parameter(
                initializers.uniform([hidden_size], -input_scaling, input_scaling),
                requires_grad=False,
            )
            if bias
            else None
        )
        self.f = getattr(torch, activation)

        self.alpha = Parameter(torch.tensor(leakage), requires_grad=False)

        self.net_gain_and_bias = net_gain_and_bias
        if net_gain_and_bias:
            self.net_a = Parameter(
                initializers.ones((hidden_size,)), requires_grad=True
            )
            self.net_b = Parameter(
                initializers.zeros((hidden_size,)), requires_grad=True
            )

        self._state_hook: Optional[Callable[..., Generator]] = None

    @torch.no_grad()
    def forward(
        self,
        input: Tensor,
        initial_state: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if initial_state is None:
            initial_state = torch.zeros(self.hidden_size).to(self.W_hat)
        _fwd_comp = (
            self._state_comp if self._state_hook is None else self._state_hook
        )

        embeddings = torch.stack(
            [state for state in _fwd_comp(input.to(self.W_hat), initial_state, mask)],  # type: ignore[operator]
            dim=0,
        )

        return embeddings

    def _state_comp(
        self, input: Tensor, initial_state: Tensor, mask: Optional[Tensor] = None
    ):
        timesteps = input.shape[0]
        state = initial_state
        for t in range(timesteps):
            in_signal_t = F.linear(
                input[t].to(self.W_in), self.W_in, self.b
            ) + F.linear(state, self.W_hat)
            if self.net_gain_and_bias:
                in_signal_t = in_signal_t * self.net_a + self.net_b
            h_t = torch.tanh(in_signal_t)
            state = (1 - self.alpha) * state + self.alpha * h_t
            yield state if mask is None else mask * state

    def register_state_hook(self, hook: Callable[..., Generator]) -> None:
        """Override the per-timestep state computation used by ``forward``.

        ``hook`` is a callable ``(input, initial_state, mask) -> generator``
        yielding one state per timestep, exactly like :meth:`_state_comp`. This
        is the supported seam for core-modifying adapters (e.g. Intrinsic
        Plasticity) to wrap the recurrence without touching private state.
        """
        self._state_hook = hook

    def clear_state_hook(self) -> None:
        """Restore the default per-timestep state computation."""
        self._state_hook = None

    @property
    def input_size(self) -> int:
        """Input dimension."""
        return self.W_in.shape[1]

    @property
    def hidden_size(self) -> int:
        """Reservoir state dimension."""
        return self.W_hat.shape[1]
