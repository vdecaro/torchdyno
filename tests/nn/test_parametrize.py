import math

import torch
from torch import nn
import torch.nn.utils.parametrize as P

from torchdyno.nn.parametrize import StableExpComplex


def test_forward_magnitude_is_strictly_stable():
    torch.manual_seed(0)
    nu = torch.randn(1000)          # any real ν
    theta = torch.randn(1000)
    lam = StableExpComplex()(nu, theta)
    assert lam.is_complex()
    assert (lam.abs() < 1.0).all()  # |λ| = exp(-exp(ν)) ∈ (0,1)


def test_right_inverse_round_trips_from_surrogates():
    torch.manual_seed(0)
    nu = torch.randn(64)
    theta = torch.rand(64) * (2 * math.pi)   # full LRU phase range [0, 2π)
    p = StableExpComplex()
    lam = p(nu, theta)
    nu2, theta2 = p.right_inverse(lam)
    # ν round-trips exactly; θ recovers only modulo 2π (torch.angle ∈ (−π, π]),
    # so verify the *eigenvalue* round-trips — the invariant that matters, since
    # θ ranges over [0, 2π) in a real LRU and drifts freely during training.
    assert torch.allclose(nu, nu2, atol=1e-5)
    assert torch.allclose(p(nu2, theta2), lam, atol=1e-6)


def test_register_parametrization_exposes_surrogates_only():
    torch.manual_seed(0)
    m = nn.Module()
    nu0, theta0 = torch.randn(8), torch.rand(8)
    m.lambda_ = nn.Parameter(torch.polar(torch.exp(-torch.exp(nu0)), theta0))
    P.register_parametrization(m, "lambda_", StableExpComplex())
    # Effective λ is complex and stable; stored parameters are the two real surrogates.
    assert m.lambda_.is_complex() and (m.lambda_.abs() < 1.0).all()
    assert all(not p.is_complex() for p in m.parameters())
    assert len(list(m.parameters())) == 2
