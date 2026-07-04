"""Reusable conformance checking for :class:`~torchdyno.models.base.SequenceCore`.

``assert_core_conforms`` runs a core over a sample input and verifies the
contract, using the core's declared :class:`CoreCapabilities` to decide which
checks apply. Model authors can call it in their own tests. Uses plain
``assert`` so importing ``torchdyno`` never pulls a test framework.
"""

from typing import Any

import torch
from torch import Tensor

from torchdyno.models.base import (
    CoreOutput,
    SequenceCore,
    UnsupportedCapability,
)


def assert_core_conforms(core: SequenceCore, x: Tensor) -> CoreOutput:
    """Assert that ``core`` satisfies the SequenceCore contract on input ``x``.

    Args:
        core: the core under test.
        x: a time-first ``(T, B, input_size)`` sample input.

    Returns:
        The :class:`CoreOutput` produced by ``core(x)``, for further assertions.

    Raises:
        AssertionError: if any declared part of the contract is violated.
    """
    caps = core.capabilities
    timesteps, batch = x.shape[0], x.shape[1]

    out = core(x)
    assert isinstance(out, CoreOutput), "forward must return a CoreOutput"

    states = out.states
    assert states.ndim == 3, f"states must be 3-D (T, B, state_size), got {states.ndim}-D"
    assert states.shape[0] == timesteps, "states time dim must match input"
    assert states.shape[1] == batch, "states batch dim must match input"
    assert states.shape[2] == core.state_size, (
        f"states width {states.shape[2]} != declared state_size {core.state_size}"
    )
    assert states.dtype == caps.dtype, (
        f"states dtype {states.dtype} != declared {caps.dtype}"
    )

    if caps.exposes_layer_states:
        assert out.layer_states is not None, "exposes_layer_states but layer_states is None"
        assert len(out.layer_states) > 0, "layer_states must be non-empty"
        for layer in out.layer_states:
            assert layer.shape[0] == timesteps, "each layer_state must have time dim T"

    if caps.supports_step:
        _assert_step_matches_forward(core, x, states)
    else:
        _assert_step_unsupported(core, x)

    if caps.differentiable:
        _assert_grad_flows(core, x)

    return out


def _assert_step_matches_forward(core: SequenceCore, x: Tensor, states: Tensor) -> None:
    stepped = []
    state: Any = None
    for t in range(x.shape[0]):
        s_t, state = core.step(x[t], state)
        stepped.append(s_t)
    stacked = torch.stack(stepped, dim=0)
    assert stacked.shape == states.shape, (
        f"step output shape {tuple(stacked.shape)} != forward {tuple(states.shape)}"
    )
    assert torch.allclose(stacked, states, atol=1e-4), (
        "stepping timestep-by-timestep must reproduce forward's states"
    )


def _assert_step_unsupported(core: SequenceCore, x: Tensor) -> None:
    raised = False
    try:
        core.step(x[0], None)
    except UnsupportedCapability:
        raised = True
    assert raised, (
        "a core that does not declare supports_step must raise "
        "UnsupportedCapability from step()"
    )


def _assert_grad_flows(core: SequenceCore, x: Tensor) -> None:
    x_req = x.detach().clone().requires_grad_(True)
    out = core(x_req)
    assert out.states.requires_grad, (
        "core declares differentiable but states do not require grad"
    )
