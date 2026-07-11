import pytest
import torch
import torch.nn.functional as F

from torchdyno.models.base import (
    CoreOutput,
    SequenceCore,
)
from torchdyno.models.ode.cornn import coRNNCore
from torchdyno.registry import (
    create_core,
    list_cores,
    render_catalog,
)
from torchdyno.testing import assert_core_conforms


def test_is_sequence_core_and_state_size():
    core = coRNNCore(input_size=2, hidden_size=8)
    assert isinstance(core, SequenceCore)
    assert core.state_size == 16  # 2 * hidden_size


def test_forward_shapes_and_final_state():
    torch.manual_seed(0)
    core = coRNNCore(input_size=3, hidden_size=8)
    out = core(torch.randn(10, 4, 3))
    assert isinstance(out, CoreOutput)
    assert out.states.shape == (10, 4, 16)
    assert out.states.dtype == torch.float32
    y, z = out.final_state
    assert y.shape == (4, 8) and z.shape == (4, 8)


def test_conforms_trainable():
    torch.manual_seed(0)
    core = coRNNCore(input_size=3, hidden_size=12)
    assert_core_conforms(core, torch.randn(9, 5, 3))


def test_update_matches_cornn_imex_equation():
    # Pins the exact symplectic update: y_n uses the UPDATED z_n.
    torch.manual_seed(0)
    core = coRNNCore(input_size=2, hidden_size=4, gamma=1.3, epsilon=0.7, dt=0.2)
    y0, z0 = torch.randn(5, 4), torch.randn(5, 4)
    xt = torch.randn(5, 2)
    emitted, (y1, z1) = core.step(xt, (y0, z0))
    pre = torch.tanh(
        F.linear(y0, core.W) + F.linear(z0, core.W_z) + F.linear(xt, core.V) + core.b
    )
    z_exp = z0 + 0.2 * (pre - 1.3 * y0 - 0.7 * z0)
    y_exp = y0 + 0.2 * z_exp
    assert torch.allclose(z1, z_exp, atol=1e-6)
    assert torch.allclose(y1, y_exp, atol=1e-6)
    assert torch.allclose(emitted, torch.cat([y_exp, z_exp], dim=-1), atol=1e-6)


def test_step_matches_forward():
    torch.manual_seed(0)
    core = coRNNCore(input_size=2, hidden_size=6)
    x = torch.randn(12, 4, 2)
    states = core(x).states
    stepped, s = [], None
    for t in range(12):
        e, s = core.step(x[t], s)
        stepped.append(e)
    assert torch.allclose(torch.stack(stepped, dim=0), states, atol=1e-5)


def test_state0_continues_recurrence():
    torch.manual_seed(0)
    core = coRNNCore(input_size=2, hidden_size=6)
    x = torch.randn(8, 3, 2)
    full = core(x).states
    first = core(x[:4])
    second = core(x[4:], state0=first.final_state)
    assert torch.allclose(second.states, full[4:], atol=1e-5)


def test_mask_zeroes_states():
    torch.manual_seed(0)
    core = coRNNCore(input_size=2, hidden_size=6)
    x = torch.randn(5, 2, 2)
    mask = torch.ones(5, 2, 1)
    mask[3:] = 0.0
    out = core(x, mask=mask)
    assert torch.count_nonzero(out.states[3:]) == 0


def test_states_bounded_over_long_horizon():
    # Oscillator is damped (epsilon>0) → states stay finite and bounded.
    torch.manual_seed(0)
    core = coRNNCore(input_size=2, hidden_size=16)
    out = core(torch.randn(1000, 2, 2))
    assert torch.isfinite(out.states).all()
    assert out.states.abs().max() < 50.0


def test_grad_flows_to_params():
    torch.manual_seed(0)
    core = coRNNCore(input_size=1, hidden_size=8)
    core(torch.randn(6, 2, 1)).states.pow(2).sum().backward()
    grads = [p.grad for p in core.parameters()]
    assert all(g is not None for g in grads) and len(grads) == 4


def test_registered_and_in_catalog():
    core = create_core("cornn", input_size=1, hidden_size=4)
    assert isinstance(core, coRNNCore)
    assert "cornn" in list_cores()
    assert "cornn" in render_catalog()


def test_rejects_bad_args():
    with pytest.raises(ValueError):
        coRNNCore(input_size=0, hidden_size=8)
    with pytest.raises(ValueError):
        coRNNCore(input_size=1, hidden_size=0)
    with pytest.raises(ValueError):
        coRNNCore(input_size=1, hidden_size=8, gamma=0.0)
    with pytest.raises(ValueError):
        coRNNCore(input_size=1, hidden_size=8, epsilon=-1.0)
    with pytest.raises(ValueError):
        coRNNCore(input_size=1, hidden_size=8, dt=0.0)


def test_reproducible_with_generator():
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    c1 = coRNNCore(input_size=2, hidden_size=8, generator=g1)
    c2 = coRNNCore(input_size=2, hidden_size=8, generator=g2)
    assert torch.equal(c1.W, c2.W)
    assert torch.equal(c1.W_z, c2.W_z)
    assert torch.equal(c1.V, c2.V)
