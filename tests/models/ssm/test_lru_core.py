import pytest
import torch

from torchdyno.models.base import CoreOutput, SequenceCore
from torchdyno.models.ssm.core import LRUCore
from torchdyno.nn.init import Ring
from torchdyno.registry import create_core, list_cores, render_catalog
from torchdyno.testing import assert_core_conforms


def test_is_sequence_core_and_state_size():
    core = LRUCore(input_size=2, n_modes=8)
    assert isinstance(core, SequenceCore)
    assert core.state_size == 16          # 2 * n_modes


def test_forward_shapes_and_final_state():
    torch.manual_seed(0)
    core = LRUCore(input_size=3, n_modes=8)
    out = core(torch.randn(10, 4, 3))
    assert isinstance(out, CoreOutput)
    assert out.states.shape == (10, 4, 16)
    assert out.states.dtype == torch.float32
    assert out.final_state.shape == (4, 8) and out.final_state.is_complex()


def test_lambda_is_stable():
    core = LRUCore(input_size=1, n_modes=64)
    assert (core.lambda_.abs() < 1.0).all()


def test_conforms_trainable():
    torch.manual_seed(0)
    core = LRUCore(input_size=3, n_modes=12)
    assert_core_conforms(core, torch.randn(9, 5, 3))


def test_scan_matches_sequential_reference():
    torch.manual_seed(0)
    core = LRUCore(input_size=2, n_modes=8)
    x = torch.randn(15, 3, 2)
    states = core(x).states
    # Reference: naive loop of the exact recurrence with the core's own params.
    lam = core.lambda_.detach()
    B = torch.complex(core.B_re, core.B_im).detach()
    gamma = torch.sqrt((1.0 - lam.abs() ** 2).clamp_min(0.0))
    Bn = gamma.unsqueeze(-1) * B
    h = torch.zeros(3, 8, dtype=lam.dtype)
    ref = []
    for t in range(15):
        h = lam * h + torch.einsum("bi,ni->bn", x[t].to(Bn.dtype), Bn)
        ref.append(torch.cat([h.real, h.imag], dim=-1))
    assert torch.allclose(states, torch.stack(ref, dim=0), atol=1e-5)


def test_step_matches_forward():
    torch.manual_seed(0)
    core = LRUCore(input_size=2, n_modes=6)
    x = torch.randn(12, 4, 2)
    states = core(x).states
    stepped, s = [], None
    for t in range(12):
        e, s = core.step(x[t], s)
        stepped.append(e)
    assert torch.allclose(torch.stack(stepped, dim=0), states, atol=1e-5)


def test_state0_continues_recurrence():
    torch.manual_seed(0)
    core = LRUCore(input_size=2, n_modes=6)
    x = torch.randn(8, 3, 2)
    full = core(x).states
    first = core(x[:4])
    second = core(x[4:], state0=first.final_state)
    assert torch.allclose(second.states, full[4:], atol=1e-5)


def test_mask_zeroes_states():
    torch.manual_seed(0)
    core = LRUCore(input_size=2, n_modes=6)
    x = torch.randn(5, 2, 2)
    mask = torch.ones(5, 2, 1)
    mask[3:] = 0.0
    out = core(x, mask=mask)
    assert torch.count_nonzero(out.states[3:]) == 0


def test_grad_flows_to_surrogates():
    torch.manual_seed(0)
    core = LRUCore(input_size=1, n_modes=8)
    core(torch.randn(6, 2, 1)).states.pow(2).sum().backward()
    grads = [p.grad for p in core.parameters()]
    assert all(g is not None for g in grads) and len(grads) == 4


def test_registered_and_in_catalog():
    core = create_core("lru", input_size=1, n_modes=4)
    assert isinstance(core, LRUCore)
    assert "lru" in list_cores()
    assert "lru" in render_catalog()


def test_rejects_bad_sizes():
    with pytest.raises(ValueError):
        LRUCore(input_size=0, n_modes=8)
    with pytest.raises(ValueError):
        LRUCore(input_size=1, n_modes=0)
