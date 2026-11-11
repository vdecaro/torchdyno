import random

import pytest
import torch
from torch import nn

from torchdyno.matrices import (
    BlockDiagonal,
    Dense,
    Diagonal,
    SkewCoupling,
    normal,
    spectral,
    uniform,
)

MATRICES = {
    "dense": lambda: Dense(6, 6),
    "diagonal": lambda: Diagonal(6, bound=torch.tanh),
    "block_diagonal": lambda: BlockDiagonal([2, 4], bound=torch.tanh),
    "skew_coupling": lambda: SkewCoupling([2, 2, 2], pairs="ring"),
}


@pytest.mark.parametrize("name", MATRICES)
def test_forward_applies_the_dense_matrix(name):
    torch.manual_seed(0)
    matrix = MATRICES[name]()
    x = torch.randn(5, 6)
    torch.testing.assert_close(matrix(x), x @ matrix.dense().T)


def test_dense_defaults_like_nn_linear():
    weight = Dense(16, 4).weight
    assert weight.shape == (4, 16)
    assert weight.abs().max() <= 16**-0.5
    assert Dense(16, 4).bias is None
    biased = Dense(16, 4, bias=nn.init.ones_)
    assert torch.equal(biased.bias, torch.ones(4))
    assert torch.equal(biased(torch.zeros(1, 16)), torch.ones(1, 4))


def test_initialisers():
    torch.manual_seed(0)
    assert Dense(50, 50, init=uniform(0.3)).weight.abs().max() <= 0.3
    assert Dense(200, 200, init=normal(0.5)).weight.std().item() == pytest.approx(
        0.5, rel=0.02
    )
    rescaled = Dense(30, 30, init=spectral(uniform(1.0), 0.9)).weight
    assert torch.linalg.eigvals(rescaled).abs().max().item() == pytest.approx(0.9)


def test_diagonal_applies_its_bound():
    diagonal = Diagonal(4, init=nn.init.ones_, bound=torch.tanh)
    torch.testing.assert_close(diagonal.diagonal(), torch.tanh(torch.ones(4)))
    assert torch.equal(Diagonal(4, init=nn.init.ones_).dense(), torch.eye(4))


def test_block_diagonal_keeps_its_zeros_through_training():
    torch.manual_seed(0)
    matrix = BlockDiagonal([3, 3], init=lambda w: nn.init.sparse_(w, 0.5))
    zeros = matrix.dense() == 0
    assert zeros[:3, 3:].all() and zeros[3:, :3].all()
    assert zeros[:3, :3].any()  # the sparse blocks have zeros inside too
    opt = torch.optim.SGD(matrix.parameters(), lr=1.0)
    matrix(torch.randn(4, 6)).pow(2).sum().backward()
    opt.step()
    assert (matrix.dense()[zeros] == 0).all()


def test_block_diagonal_blocks_are_orthogonal_by_default():
    blocks = BlockDiagonal([3, 5]).dense()
    torch.testing.assert_close(blocks[:3, :3] @ blocks[:3, :3].T, torch.eye(3))
    torch.testing.assert_close(blocks[3:, 3:] @ blocks[3:, 3:].T, torch.eye(5))


def test_skew_coupling_stays_skew_and_sparse_through_training():
    torch.manual_seed(0)
    coupling = SkewCoupling([2, 2, 2], pairs=[(0, 1)])
    uncoupled = coupling.dense() == 0
    opt = torch.optim.SGD(coupling.parameters(), lr=0.1)
    coupling(torch.randn(4, 6)).pow(2).sum().backward()
    opt.step()
    coupled = coupling.dense()
    torch.testing.assert_close(coupled + coupled.T, torch.zeros(6, 6))
    assert (coupled[uncoupled] == 0).all()


def _coupled_pairs(coupling, sizes):
    offsets = [0, *torch.tensor(sizes).cumsum(0).tolist()]
    dense = coupling.dense()
    return {
        (i, j)
        for i in range(len(sizes))
        for j in range(i + 1, len(sizes))
        if dense[offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]].any()
    }


def test_skew_coupling_pair_forms():
    sizes = [2, 2, 2, 2]  # 6 possible pairs
    torch.manual_seed(0)
    assert len(_coupled_pairs(SkewCoupling(sizes, pairs=1), sizes)) == 1
    assert len(_coupled_pairs(SkewCoupling(sizes, pairs=4), sizes)) == 4
    assert len(_coupled_pairs(SkewCoupling(sizes, pairs=0.5), sizes)) == 3
    ring = _coupled_pairs(SkewCoupling(sizes, pairs="ring"), sizes)
    assert ring == {(0, 1), (1, 2), (2, 3), (0, 3)}
    assert _coupled_pairs(SkewCoupling(sizes, pairs=[(2, 0)]), sizes) == {(0, 2)}


def test_skew_coupling_rejects_self_pairs():
    with pytest.raises(ValueError, match="coupled with itself"):
        SkewCoupling([2, 2], pairs=[(1, 1)])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matrices_keep_their_dtype(dtype):
    for matrix in (
        Dense(3, 3, bias=nn.init.zeros_, dtype=dtype),
        Diagonal(3, dtype=dtype),
        BlockDiagonal([3], dtype=dtype),
        SkewCoupling([2, 1], pairs="ring", dtype=dtype),
    ):
        assert matrix(torch.randn(2, 3, dtype=dtype)).dtype == dtype


def test_complex_dense_takes_real_input():
    torch.manual_seed(0)
    matrix = Dense(3, 4, init=normal(0.5), bias=normal(0.1), dtype=torch.cdouble)
    x = torch.randn(5, 3, dtype=torch.float64)
    out = matrix(x)
    assert out.dtype == torch.cdouble
    bias = torch.view_as_complex(matrix.bias)
    torch.testing.assert_close(out, x.to(torch.cdouble) @ matrix.dense().T + bias)


@pytest.mark.parametrize(
    "make",
    [
        lambda: Dense(3, 4, bias=normal(0.1), dtype=torch.cfloat),
        lambda: Diagonal(3, dtype=torch.cfloat),
    ],
)
@pytest.mark.parametrize("cast", [lambda m: m.double(), lambda m: m.to(torch.float64)])
def test_complex_matrices_keep_both_parts_under_casts(make, cast):
    torch.manual_seed(0)
    matrix = make()
    x = torch.randn(5, 3)
    expected = matrix(x).to(torch.cdouble)
    cast(matrix)
    out = matrix(x.double())
    assert out.dtype == torch.cdouble
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)


def test_spectral_rejects_what_it_cannot_rescale():
    with pytest.raises(ValueError, match="rho"):
        spectral(uniform(1.0), 0.0)
    delay_line = spectral(lambda w: w.copy_(torch.diag(torch.ones(3), -1)), 0.9)
    with pytest.raises(ValueError, match="eigenvalue"):
        delay_line(torch.empty(4, 4))


def test_skew_coupling_draws_its_pairs_from_torch():
    masks = []
    for python_seed in (0, 1):
        random.seed(python_seed)
        torch.manual_seed(0)
        masks.append(SkewCoupling([2] * 8, pairs=5).mask)
    assert torch.equal(*masks)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matrices_initialise_in_half_precision(dtype):
    for matrix in (
        Dense(3, 3, init=spectral(uniform(1.0), 0.9), dtype=dtype),
        BlockDiagonal([3], dtype=dtype),
        SkewCoupling([2, 1], pairs="ring", dtype=dtype),
    ):
        assert matrix.weight.dtype == dtype
        assert matrix.dense().isfinite().all()


def test_complex_initialisers_draw_both_parts():
    torch.manual_seed(0)
    weight = Dense(300, 300, init=normal(0.5), dtype=torch.cfloat).dense()
    for part in (weight.real, weight.imag):
        assert part.std().item() == pytest.approx(0.5, rel=0.02)
    bounded = Dense(50, 50, init=uniform(0.3), dtype=torch.cfloat).dense()
    assert bounded.real.abs().max() <= 0.3 and bounded.imag.abs().max() <= 0.3
    assert bounded.imag.abs().max() > 0.25
