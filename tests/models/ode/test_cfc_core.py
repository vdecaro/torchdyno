import pytest
import torch

from torchdyno.models.base import (
    CoreOutput,
    SequenceCore,
)
from torchdyno.models.ode.cfc import CfCCore
from torchdyno.registry import (
    create_core,
    list_cores,
    render_catalog,
)
from torchdyno.testing import assert_core_conforms


def test_is_sequence_core_and_state_size():
    core = CfCCore(input_size=2, hidden_size=8)
    assert isinstance(core, SequenceCore)
    assert core.state_size == 8  # == hidden_size (single state)


def test_forward_shapes_and_final_state():
    torch.manual_seed(0)
    core = CfCCore(input_size=3, hidden_size=8)
    out = core(torch.randn(10, 4, 3))
    assert isinstance(out, CoreOutput)
    assert out.states.shape == (10, 4, 8)
    assert out.states.dtype == torch.float32
    assert out.final_state.shape == (4, 8)


def test_conforms_trainable():
    torch.manual_seed(0)
    assert_core_conforms(CfCCore(input_size=3, hidden_size=12), torch.randn(9, 5, 3))


def test_conforms_no_backbone():
    torch.manual_seed(0)
    core = CfCCore(input_size=3, hidden_size=10, backbone_layers=0)
    assert_core_conforms(core, torch.randn(7, 4, 3))


def test_update_matches_cfc_gated_blend():
    # Pins the exact gated blend and dt placement.
    torch.manual_seed(0)
    core = CfCCore(input_size=2, hidden_size=4, backbone_layers=1, dt=0.5)
    x0 = torch.randn(5, 4)
    u = torch.randn(5, 2)
    emitted, x1 = core.step(u, x0)
    b = core.backbone(torch.cat([x0, u], dim=-1))
    ff1 = torch.tanh(core.head_ff1(b))
    ff2 = torch.tanh(core.head_ff2(b))
    gate = torch.sigmoid(core.head_ta(b) * 0.5 + core.head_tb(b))
    expected = ff1 * (1.0 - gate) + ff2 * gate
    assert torch.allclose(x1, expected, atol=1e-6)
    assert torch.allclose(emitted, expected, atol=1e-6)


def test_step_matches_forward():
    torch.manual_seed(0)
    core = CfCCore(input_size=2, hidden_size=6)
    x = torch.randn(12, 4, 2)
    states = core(x).states
    stepped, s = [], None
    for t in range(12):
        e, s = core.step(x[t], s)
        stepped.append(e)
    assert torch.allclose(torch.stack(stepped, dim=0), states, atol=1e-5)


def test_state0_continues_recurrence():
    torch.manual_seed(0)
    core = CfCCore(input_size=2, hidden_size=6)
    x = torch.randn(8, 3, 2)
    full = core(x).states
    first = core(x[:4])
    second = core(x[4:], state0=first.final_state)
    assert torch.allclose(second.states, full[4:], atol=1e-5)


def test_mask_zeroes_states():
    torch.manual_seed(0)
    core = CfCCore(input_size=2, hidden_size=6)
    x = torch.randn(5, 2, 2)
    mask = torch.ones(5, 2, 1)
    mask[3:] = 0.0
    out = core(x, mask=mask)
    assert torch.count_nonzero(out.states[3:]) == 0


def test_states_bounded_in_unit_interval():
    # x_new = ff1*(1-gate)+ff2*gate, |ff1|,|ff2|<=1, gate in (0,1) => |x|<=1.
    torch.manual_seed(0)
    core = CfCCore(input_size=2, hidden_size=16)
    out = core(torch.randn(1000, 2, 2))
    assert torch.isfinite(out.states).all()
    assert out.states.abs().max() <= 1.0


def test_grad_flows_to_all_params():
    torch.manual_seed(0)
    core = CfCCore(input_size=1, hidden_size=8)
    core(torch.randn(6, 2, 1)).states.pow(2).sum().backward()
    grads = [p.grad for p in core.parameters()]
    assert len(grads) > 0 and all(g is not None for g in grads)


def test_reproducible_with_generator():
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    c1 = CfCCore(input_size=2, hidden_size=8, generator=g1)
    c2 = CfCCore(input_size=2, hidden_size=8, generator=g2)
    for p1, p2 in zip(c1.parameters(), c2.parameters()):
        assert torch.equal(p1, p2)


def test_registered_and_in_catalog():
    core = create_core("cfc", input_size=1, hidden_size=4)
    assert isinstance(core, CfCCore)
    assert "cfc" in list_cores()
    assert "cfc" in render_catalog()


def test_rejects_bad_args():
    with pytest.raises(ValueError):
        CfCCore(input_size=0, hidden_size=8)
    with pytest.raises(ValueError):
        CfCCore(input_size=1, hidden_size=0)
    with pytest.raises(ValueError):
        CfCCore(input_size=1, hidden_size=8, backbone_layers=-1)
    with pytest.raises(ValueError):
        CfCCore(input_size=1, hidden_size=8, dt=0.0)
    with pytest.raises(ValueError):
        CfCCore(input_size=1, hidden_size=8, activation="nope")
