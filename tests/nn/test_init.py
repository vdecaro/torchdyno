import math

import torch

from torchdyno.nn.init import (
    Ring,
    S4DInv,
    S4DLin,
)


def test_magnitudes_and_phases_within_ring():
    torch.manual_seed(0)
    nu, theta = Ring(r_min=0.4, r_max=0.9, max_phase=math.pi)(5000)
    mag = torch.exp(-torch.exp(nu))
    assert (mag >= 0.4 - 1e-4).all() and (mag <= 0.9 + 1e-4).all()
    assert (theta >= 0.0).all() and (theta <= math.pi + 1e-6).all()


def test_returns_requested_shape():
    nu, theta = Ring()(32)
    assert nu.shape == (32,) and theta.shape == (32,)


def test_deterministic_under_generator():
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    n1, t1 = Ring()(16, generator=g1)
    n2, t2 = Ring()(16, generator=g2)
    assert torch.equal(n1, n2) and torch.equal(t1, t2)


def test_rejects_bad_bounds():
    import pytest

    with pytest.raises(ValueError):
        Ring(r_min=0.9, r_max=0.5)


def test_s4dlin_A_placement():
    log_are, a_im, _ = S4DLin()(16)
    A = torch.complex(-torch.exp(log_are), a_im)
    assert torch.allclose(A.real, torch.full((16,), -0.5), atol=1e-6)
    n = torch.arange(16, dtype=torch.float32)
    assert torch.allclose(A.imag, math.pi * n, atol=1e-5)


def test_s4dinv_A_placement():
    N = 16
    log_are, a_im, _ = S4DInv()(N)
    assert torch.allclose(-torch.exp(log_are), torch.full((N,), -0.5), atol=1e-6)
    n = torch.arange(N, dtype=torch.float32)
    expected = (N / math.pi) * (N / (2 * n + 1) - 1)
    assert torch.allclose(a_im, expected, atol=1e-4)


def test_s4d_dt_within_bounds():
    _, _, log_dt = S4DLin(dt_min=1e-3, dt_max=1e-1)(5000)
    dt = torch.exp(log_dt)
    assert (dt >= 1e-3 - 1e-6).all() and (dt <= 1e-1 + 1e-6).all()


def test_s4d_deterministic_under_generator():
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    a = S4DLin()(16, generator=g1)
    b = S4DLin()(16, generator=g2)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_s4d_rejects_bad_dt():
    import pytest

    with pytest.raises(ValueError):
        S4DLin(dt_min=0.1, dt_max=0.01)
