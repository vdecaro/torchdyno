import pytest
import torch
from torch import nn

from torchdyno.models.base import (
    CoreCapabilities,
    CoreOutput,
    SequenceCore,
    UnsupportedCapability,
)


def _caps(**overrides):
    base = dict(
        compute_mode="loop",
        differentiable=False,
        trainable_recurrence=False,
        supports_step=False,
        admits_feedback=False,
        exposes_layer_states=False,
    )
    base.update(overrides)
    return CoreCapabilities(**base)


class _MinimalCore(SequenceCore):
    def __init__(self):
        super().__init__()
        self.input_size = 3
        self.state_size = 3
        self.capabilities = _caps()

    def forward(self, x, *, state0=None, mask=None):
        return CoreOutput(states=x, final_state=x[-1])


def test_capabilities_defaults_dtype_to_float32():
    caps = _caps()
    assert caps.dtype == torch.float32


def test_capabilities_is_frozen():
    caps = _caps()
    with pytest.raises(Exception):
        caps.compute_mode = "scan"  # frozen dataclass -> error


def test_core_output_layer_states_default_none():
    out = CoreOutput(states=torch.zeros(2, 1, 3), final_state=None)
    assert out.layer_states is None


def test_sequence_core_is_abstract():
    with pytest.raises(TypeError):
        SequenceCore()  # abstract forward -> cannot instantiate


def test_minimal_core_forward_returns_core_output():
    core = _MinimalCore()
    x = torch.randn(4, 2, 3)
    out = core(x)
    assert isinstance(out, CoreOutput)
    assert torch.equal(out.states, x)


def test_step_raises_unsupported_by_default():
    core = _MinimalCore()
    with pytest.raises(UnsupportedCapability):
        core.step(torch.randn(2, 3), None)


def test_is_nn_module():
    assert isinstance(_MinimalCore(), nn.Module)
