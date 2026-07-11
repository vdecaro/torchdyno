import torch

from torchdyno.models.ode.cornn import coRNNCore
from torchdyno.testing import assert_core_conforms


def test_frozen_params_have_no_grad():
    core = coRNNCore(input_size=2, hidden_size=8, trainable=False)
    assert all(not p.requires_grad for p in core.parameters())


def test_frozen_capabilities_flipped():
    core = coRNNCore(input_size=2, hidden_size=8, trainable=False)
    assert core.capabilities.differentiable is False
    assert core.capabilities.trainable_recurrence is False
    assert core.capabilities.compute_mode == "loop"
    assert core.capabilities.supports_step is True


def test_frozen_conforms():
    torch.manual_seed(0)
    core = coRNNCore(input_size=3, hidden_size=10, trainable=False)
    assert_core_conforms(core, torch.randn(8, 4, 3))


def test_frozen_states_finite_long_horizon():
    torch.manual_seed(0)
    core = coRNNCore(input_size=1, hidden_size=32, trainable=False)
    out = core(torch.randn(500, 2, 1))
    assert torch.isfinite(out.states).all()
    assert out.states.abs().max() < 50.0


def test_trainable_default_is_true():
    core = coRNNCore(input_size=1, hidden_size=4)
    assert core.capabilities.trainable_recurrence is True
    assert all(p.requires_grad for p in core.parameters())
