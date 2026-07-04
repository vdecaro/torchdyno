import torch

from torchdyno.models.assembly import build_coupling, SkewAntisymmetricCoupling


def test_returns_skew_coupling_module():
    coupling = build_coupling([4, 4], coupling_topology=[(0, 1)])
    assert isinstance(coupling, SkewAntisymmetricCoupling)


def test_coupling_matrix_is_skew_symmetric():
    torch.manual_seed(0)
    coupling = build_coupling([3, 3, 3], coupling_topology=[(0, 1), (1, 2)])
    L = coupling.couplings  # (H, H)
    assert L.shape == (9, 9)
    assert torch.allclose(L + L.T, torch.zeros_like(L), atol=1e-6)


def test_coupling_forward_maps_state_width():
    torch.manual_seed(0)
    coupling = build_coupling([4, 4], coupling_topology=[(0, 1)])
    v = torch.randn(5, 8)  # (B, H)
    out = coupling(v)
    assert out.shape == (5, 8)


def test_diagonal_blocks_are_zero():
    # A skew off-block-diagonal coupling must not couple a module to itself.
    torch.manual_seed(0)
    coupling = build_coupling([2, 2], coupling_topology=[(0, 1)])
    L = coupling.couplings
    assert torch.allclose(L[:2, :2], torch.zeros(2, 2), atol=1e-6)
    assert torch.allclose(L[2:, 2:], torch.zeros(2, 2), atol=1e-6)


def test_accepts_callable_initializer():
    calls = {"n": 0}

    def init_fn(shape, dtype=torch.float32):
        calls["n"] += 1
        return torch.ones(shape, dtype=dtype)

    coupling = build_coupling([2, 2], coupling_topology=[(0, 1)], coupling_block_init=init_fn)
    assert calls["n"] == 1  # one block for the single pair
    assert isinstance(coupling, SkewAntisymmetricCoupling)


def test_accepts_non_orthogonal_string_initializer():
    # A str initializer whose 2nd positional arg is NOT dtype (e.g. "diagonal"
    # has min_val there) must still work because dtype is passed by keyword.
    coupling = build_coupling([3, 3], coupling_topology=[(0, 1)], coupling_block_init="diagonal")
    L = coupling.couplings
    assert torch.allclose(L + L.T, torch.zeros_like(L), atol=1e-6)
