"""The SequenceCore contract: the interface every recurrent model satisfies.

A core consumes a time-first ``(T, B, input_size)`` sequence and emits a
``(T, B, state_size)`` state sequence via ``forward``. Everything beyond that
minimal surface is either optional (``step``) or advertised through
``CoreCapabilities`` so that tooling can adapt to each core without inspecting
its internals.
"""

from abc import (
    ABC,
    abstractmethod,
)
from dataclasses import dataclass
from typing import (
    Any,
    List,
    Literal,
    Optional,
    Tuple,
)

import torch
from torch import (
    Tensor,
    nn,
)


class UnsupportedCapability(Exception):
    """Raised when a core is asked for a capability it does not declare."""


@dataclass(frozen=True)
class CoreCapabilities:
    """What a :class:`SequenceCore` can do, declared rather than inferred.

    Args:
        compute_mode: how ``forward`` is computed internally.
        differentiable: whether autograd flows through ``forward``.
        trainable_recurrence: whether the recurrence has trainable parameters
            (``False`` for frozen reservoirs).
        supports_step: whether ``step`` is available for streaming/inference.
        admits_feedback: whether head output can be routed back into the input.
        exposes_layer_states: whether ``CoreOutput.layer_states`` is populated.
        dtype: the dtype of the emitted states.
    """

    compute_mode: Literal["loop", "scan", "solver", "fixed_point"]
    differentiable: bool
    trainable_recurrence: bool
    supports_step: bool
    admits_feedback: bool
    exposes_layer_states: bool
    dtype: torch.dtype = torch.float32


@dataclass
class CoreOutput:
    """The result of running a core over a sequence.

    Args:
        states: the emitted state sequence, shape ``(T, B, state_size)``.
        final_state: opaque state usable to continue the recurrence.
        layer_states: per-layer state sequences, populated iff the core
            declares ``exposes_layer_states``.
    """

    states: Tensor
    final_state: Any
    layer_states: Optional[List[Tensor]] = None


class SequenceCore(nn.Module, ABC):
    """Base class for recurrent cores.

    Subclasses must set ``input_size``, ``state_size``, and ``capabilities`` in
    their ``__init__`` and implement ``forward``. ``step`` is optional and
    should only be overridden when ``capabilities.supports_step`` is ``True``.
    """

    input_size: int
    state_size: int
    capabilities: CoreCapabilities

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        *,
        state0: Any = None,
        mask: Optional[Tensor] = None,
    ) -> CoreOutput:
        """Run the core over a ``(T, B, input_size)`` sequence."""
        raise NotImplementedError

    def step(self, x_t: Tensor, state: Any) -> Tuple[Tensor, Any]:
        """Advance the recurrence by one timestep.

        Args:
            x_t: a single-timestep input of shape ``(B, input_size)``.
            state: the state returned by a previous ``step`` (``None`` starts
                from the zero state).

        Returns:
            A tuple ``(emitted_state, new_state)`` where ``emitted_state`` has
            shape ``(B, state_size)``.
        """
        raise UnsupportedCapability(
            f"{type(self).__name__} does not support step()."
        )
