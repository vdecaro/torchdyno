import pytest
import torch

from torchdyno.models.base import CoreCapabilities, CoreOutput, SequenceCore
from torchdyno.testing import assert_core_conforms


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


class EchoCore(SequenceCore):
    """A valid identity core (emitted state == input) whose ``step`` honours
    its declared capability: when it does NOT declare support it delegates to
    the raising base ``step``."""

    def __init__(self, supports_step=True, exposes_layer_states=False):
        super().__init__()
        self.input_size = 3
        self.state_size = 3
        self.capabilities = _caps(
            supports_step=supports_step,
            exposes_layer_states=exposes_layer_states,
        )

    def forward(self, x, *, state0=None, mask=None):
        layer_states = [x] if self.capabilities.exposes_layer_states else None
        return CoreOutput(states=x, final_state=x[-1], layer_states=layer_states)

    def step(self, x_t, state):
        if not self.capabilities.supports_step:
            return super().step(x_t, state)  # inherits UnsupportedCapability
        return x_t, None


class LyingStepCore(SequenceCore):
    """Declares supports_step=False but its ``step`` does NOT raise — the
    checker must reject this inconsistency."""

    def __init__(self):
        super().__init__()
        self.input_size = 3
        self.state_size = 3
        self.capabilities = _caps(supports_step=False)

    def forward(self, x, *, state0=None, mask=None):
        return CoreOutput(states=x, final_state=x[-1])

    def step(self, x_t, state):
        return x_t, None


class WrongShapeCore(SequenceCore):
    """Declares state_size 3 but emits width 2 — must be rejected."""

    def __init__(self):
        super().__init__()
        self.input_size = 3
        self.state_size = 3
        self.capabilities = _caps()

    def forward(self, x, *, state0=None, mask=None):
        return CoreOutput(states=x[..., :2], final_state=None)


def test_conforms_returns_core_output_for_valid_core():
    core = EchoCore(supports_step=True)
    x = torch.randn(5, 2, 3)
    out = assert_core_conforms(core, x)
    assert isinstance(out, CoreOutput)


def test_conforms_non_stepping_core():
    # supports_step=False AND step() raises (delegates to base) -> valid.
    core = EchoCore(supports_step=False)
    x = torch.randn(5, 2, 3)
    assert_core_conforms(core, x)


def test_rejects_wrong_state_width():
    core = WrongShapeCore()
    x = torch.randn(5, 2, 3)
    with pytest.raises(AssertionError):
        assert_core_conforms(core, x)


def test_checks_step_agreement_when_supported():
    core = EchoCore(supports_step=True)
    x = torch.randn(5, 2, 3)
    assert_core_conforms(core, x)  # identity step reproduces states


def test_rejects_lying_step_core():
    # Declares supports_step=False but step() does not raise -> rejected.
    core = LyingStepCore()
    x = torch.randn(5, 2, 3)
    with pytest.raises(AssertionError):
        assert_core_conforms(core, x)


def test_checks_layer_states_when_exposed():
    core = EchoCore(supports_step=True, exposes_layer_states=True)
    x = torch.randn(5, 2, 3)
    out = assert_core_conforms(core, x)
    assert out.layer_states is not None and len(out.layer_states) == 1
