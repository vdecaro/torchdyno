import torch

from torchdyno.models.ssm.lru import LRUCore
from torchdyno.testing import assert_core_conforms


def test_frozen_params_have_no_grad():
    core = LRUCore(input_size=2, n_modes=8, trainable=False)
    assert all(not p.requires_grad for p in core.parameters())


def test_frozen_capabilities_flipped():
    core = LRUCore(input_size=2, n_modes=8, trainable=False)
    assert core.capabilities.differentiable is False
    assert core.capabilities.trainable_recurrence is False
    assert core.capabilities.compute_mode == "scan"
    assert core.capabilities.supports_step is True


def test_frozen_conforms():
    torch.manual_seed(0)
    core = LRUCore(input_size=3, n_modes=10, trainable=False)
    assert_core_conforms(core, torch.randn(8, 4, 3))


def test_frozen_states_are_finite_and_stable_long_horizon():
    torch.manual_seed(0)
    core = LRUCore(input_size=1, n_modes=32, trainable=False)
    out = core(torch.randn(500, 2, 1))
    assert torch.isfinite(out.states).all()


def test_trainable_default_is_true():
    core = LRUCore(input_size=1, n_modes=4)
    assert core.capabilities.trainable_recurrence is True
    assert all(p.requires_grad for p in core.parameters())
