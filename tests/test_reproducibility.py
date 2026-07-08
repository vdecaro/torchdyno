import torch

from torchdyno import get_rng_state, seed_all, set_rng_state, SequenceModel
from torchdyno.heads import RegressionHead
from torchdyno.models.esn import ESNCore
from torchdyno.optim import RidgeRegression


def test_seed_all_makes_runs_bit_identical():
    seed_all(123)
    a = torch.randn(50)
    seed_all(123)
    b = torch.randn(50)
    assert torch.equal(a, b)


def test_seed_all_seeds_model_construction():
    seed_all(7)
    w1 = ESNCore(input_size=1, layer_sizes=[16]).reservoirs[0].W_hat.clone()
    seed_all(7)
    w2 = ESNCore(input_size=1, layer_sizes=[16]).reservoirs[0].W_hat.clone()
    assert torch.equal(w1, w2)


def test_rng_state_round_trip():
    state = get_rng_state()
    a = torch.randn(10)
    set_rng_state(state)
    b = torch.randn(10)
    assert torch.equal(a, b)
    assert {"python", "numpy", "torch"} <= set(state)


def test_ridge_fit_captures_rng_in_fitresult():
    torch.manual_seed(0)
    core = ESNCore(input_size=1, layer_sizes=[32])
    model = SequenceModel(core, RegressionHead(core.state_size, 1))
    x = torch.randn(30, 4, 1)
    result = RidgeRegression(l2=1e-6).fit(model, [(x, x.clone())])
    assert result.rng is not None
    assert {"python", "numpy", "torch"} <= set(result.rng)
