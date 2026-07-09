import math

import torch

from torchdyno.nn.init import Ring


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
