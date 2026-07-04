import torch

from torchdyno.models.assembly.adadiag import FrequencyGate


def test_output_shape():
    gate = FrequencyGate(input_size=2, hidden_size=6)
    nu = gate(torch.randn(4, 2), torch.randn(4, 6))
    assert nu.shape == (4, 6)


def test_values_bounded_even_for_extreme_logits():
    torch.manual_seed(0)
    gate = FrequencyGate(input_size=2, hidden_size=6, lo=1e-5, hi=0.2)
    # Saturating logits drive float32 sigmoid to exactly 0.0/1.0, so ν may
    # equal a bound exactly — the guarantee is the CLOSED interval [lo, hi].
    nu = gate(torch.randn(10, 2) * 100, torch.randn(10, 6) * 100)
    assert (nu >= 1e-5).all() and (nu <= 0.2).all()
    # For moderate logits it is strictly interior:
    nu2 = gate(torch.randn(10, 2), torch.randn(10, 6))
    assert (nu2 > 1e-5).all() and (nu2 < 0.2).all()


def test_gate_state_initialized_to_zero():
    gate = FrequencyGate(input_size=2, hidden_size=6)
    assert torch.equal(gate.gate_state, torch.zeros(6))


def test_state_independent_at_init():
    # With gate_state==0, ν must not depend on v.
    torch.manual_seed(0)
    gate = FrequencyGate(input_size=2, hidden_size=6)
    x = torch.randn(4, 2)
    nu_a = gate(x, torch.randn(4, 6))
    nu_b = gate(x, torch.randn(4, 6))
    assert torch.allclose(nu_a, nu_b)


def test_grad_flows():
    gate = FrequencyGate(input_size=2, hidden_size=6)
    x = torch.randn(4, 2, requires_grad=True)
    nu = gate(x, torch.randn(4, 6))
    nu.sum().backward()
    assert gate.gate_input.weight.grad is not None
