import pytest
import torch

from torchdyno.models.base import (
    CoreOutput,
    SequenceCore,
)
from torchdyno.models.ssm.s4d import S4DCore
from torchdyno.registry import (
    create_core,
    list_cores,
    render_catalog,
)
from torchdyno.testing import assert_core_conforms


def test_is_sequence_core_and_state_size():
    core = S4DCore(input_size=2, n_modes=8)
    assert isinstance(core, SequenceCore)
    assert core.state_size == 16


def test_forward_shapes_and_final_state():
    torch.manual_seed(0)
    core = S4DCore(input_size=3, n_modes=8)
    out = core(torch.randn(10, 4, 3))
    assert isinstance(out, CoreOutput)
    assert out.states.shape == (10, 4, 16)
    assert out.states.dtype == torch.float32
    assert out.final_state.shape == (4, 8) and out.final_state.is_complex()


def test_A_has_negative_real_part():
    core = S4DCore(input_size=1, n_modes=32)
    assert (core.A.real < 0).all()


def test_discrete_coeff_is_stable():
    core = S4DCore(input_size=1, n_modes=32)
    a_diag, _ = core._recurrence()
    assert (a_diag.abs() < 1.0).all()


def test_conforms_trainable():
    torch.manual_seed(0)
    core = S4DCore(input_size=3, n_modes=12)
    assert_core_conforms(core, torch.randn(9, 5, 3))


def test_conforms_frozen():
    torch.manual_seed(0)
    core = S4DCore(input_size=3, n_modes=12, trainable=False)
    assert_core_conforms(core, torch.randn(9, 5, 3))
    assert all(not p.requires_grad for p in core.parameters())
    assert core.capabilities.differentiable is False


def test_scan_matches_sequential_reference():
    torch.manual_seed(0)
    core = S4DCore(input_size=2, n_modes=8)
    x = torch.randn(15, 3, 2)
    states = core(x).states
    a_diag, B = core._recurrence()
    a_diag, B = a_diag.detach(), B.detach()
    h = torch.zeros(3, 8, dtype=a_diag.dtype)
    ref = []
    for t in range(15):
        h = a_diag * h + torch.einsum("bi,ni->bn", x[t].to(B.dtype), B)
        ref.append(torch.cat([h.real, h.imag], dim=-1))
    assert torch.allclose(states, torch.stack(ref, dim=0), atol=1e-5)


def test_step_matches_forward():
    torch.manual_seed(0)
    core = S4DCore(input_size=2, n_modes=6)
    x = torch.randn(12, 4, 2)
    states = core(x).states
    stepped, s = [], None
    for t in range(12):
        e, s = core.step(x[t], s)
        stepped.append(e)
    assert torch.allclose(torch.stack(stepped, dim=0), states, atol=1e-5)


def test_state0_continues_recurrence():
    torch.manual_seed(0)
    core = S4DCore(input_size=2, n_modes=6)
    x = torch.randn(8, 3, 2)
    full = core(x).states
    first = core(x[:4])
    second = core(x[4:], state0=first.final_state)
    assert torch.allclose(second.states, full[4:], atol=1e-5)


def test_grad_flows_to_surrogates():
    torch.manual_seed(0)
    core = S4DCore(input_size=1, n_modes=8)
    core(torch.randn(6, 2, 1)).states.pow(2).sum().backward()
    grads = [p.grad for p in core.parameters()]
    assert all(g is not None for g in grads) and len(grads) == 5


def test_init_variants_differ():
    lin = S4DCore(input_size=1, n_modes=16, init="s4d-lin")
    inv = S4DCore(input_size=1, n_modes=16, init="s4d-inv")
    assert not torch.allclose(lin.A.imag, inv.A.imag)


def test_registered_and_in_catalog():
    core = create_core("s4d", input_size=1, n_modes=4)
    assert isinstance(core, S4DCore)
    assert "s4d" in list_cores()
    assert "s4d" in render_catalog()


def test_rejects_bad_sizes():
    with pytest.raises(ValueError):
        S4DCore(input_size=0, n_modes=8)
    with pytest.raises(ValueError):
        S4DCore(input_size=1, n_modes=0)
