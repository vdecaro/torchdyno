import torch
import torch.nn.functional as F

from torchdyno.models.assembly.adadiag import AdaDiagCore
from torchdyno.models.base import CoreOutput, SequenceCore
from torchdyno.testing import assert_core_conforms


def _core(**kw):
    params = dict(
        input_size=2,
        block_sizes=[3, 3],
        coupling_topology=[(0, 1)],
    )
    params.update(kw)
    return AdaDiagCore(**params)


def test_is_sequence_core():
    assert isinstance(_core(), SequenceCore)


def test_state_size_is_sum_of_blocks():
    assert _core(block_sizes=[4, 5, 6]).state_size == 15


def test_forward_shape_and_output():
    torch.manual_seed(0)
    core = _core(block_sizes=[8, 8])
    out = core(torch.randn(10, 4, 2))
    assert isinstance(out, CoreOutput)
    assert out.states.shape == (10, 4, 16)


def test_conforms():
    torch.manual_seed(0)
    core = _core(block_sizes=[6, 6])
    assert_core_conforms(core, torch.randn(7, 3, 2))


def test_update_matches_adadiag_equation():
    # Pins the exact fused update: coupling AND input are inside the ν scaling.
    torch.manual_seed(0)
    core = _core(block_sizes=[2, 2], activation="relu")
    v0 = torch.randn(3, core.state_size)
    xt = torch.randn(3, 2)

    v1, _ = core.step(xt, v0)

    wphi = torch.tanh(core.beta_W) * torch.relu(v0)
    lv = core.couplings(v0)
    bu = F.linear(xt, core.input_mat)
    nu = core.gate(xt, v0)
    expected = v0 + nu * (-v0 + wphi + lv + bu)
    assert torch.allclose(v1, expected, atol=1e-6)


def test_nu_bounds_respected_in_update():
    torch.manual_seed(0)
    core = _core(gate_bounds=(1e-5, 0.2))
    xt = torch.randn(3, 2)
    v0 = torch.randn(3, core.state_size)
    nu = core.gate(xt, v0)
    assert (nu > 1e-5).all() and (nu < 0.2).all()


def test_states_bounded_over_long_sequence():
    # Contractive-by-design: states must not blow up over a long horizon.
    torch.manual_seed(0)
    core = _core(block_sizes=[8, 8])
    out = core(torch.randn(500, 2, 2))
    assert torch.isfinite(out.states).all()
    assert out.states.abs().max() < 50.0


def test_beta_w_effective_weight_is_bounded():
    core = _core()
    assert (torch.tanh(core.beta_W).abs() < 1.0).all()
