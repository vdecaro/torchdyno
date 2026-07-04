import torch
import torch.nn.functional as F

from torchdyno.models.assembly.scn import SCNCore
from torchdyno.models.base import CoreOutput, SequenceCore
from torchdyno.testing import assert_core_conforms


def _core(**kw):
    params = dict(input_size=2, block_sizes=[3, 3], coupling_topology=[(0, 1)])
    params.update(kw)
    return SCNCore(**params)


def test_is_sequence_core():
    assert isinstance(_core(), SequenceCore)


def test_state_size_is_sum_of_blocks():
    assert _core(block_sizes=[4, 5]).state_size == 9


def test_forward_shape():
    torch.manual_seed(0)
    out = _core(block_sizes=[8, 8])(torch.randn(10, 4, 2))
    assert isinstance(out, CoreOutput)
    assert out.states.shape == (10, 4, 16)


def test_conforms():
    torch.manual_seed(0)
    assert_core_conforms(_core(block_sizes=[6, 6]), torch.randn(7, 3, 2))


def test_update_matches_euler_equation():
    torch.manual_seed(0)
    core = _core(block_sizes=[2, 2], eul_step=0.1, activation="relu")
    v0 = torch.randn(3, core.state_size)
    xt = torch.randn(3, 2)
    v1, _ = core.step(xt, v0)
    expected = v0 + 0.1 * (
        -v0
        + core.blocks(torch.relu(v0))
        + core.couplings(v0)
        + F.linear(xt, core.input_mat)
    )
    assert torch.allclose(v1, expected, atol=1e-6)


def test_fixed_blocks_are_frozen():
    core = _core(constrained_blocks="fixed")
    # The block-diagonal weight is frozen; only couplings/input are trainable.
    assert not core.blocks._blocks.requires_grad


def test_states_finite_over_long_sequence():
    torch.manual_seed(0)
    out = _core(block_sizes=[8, 8])(torch.randn(300, 2, 2))
    assert torch.isfinite(out.states).all()
