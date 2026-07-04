import torch

import torchdyno
from torchdyno.heads import RegressionHead
from torchdyno.model import SequenceModel
from torchdyno.models.esn.core import ESNCore
from torchdyno.optim.ridge import RidgeRegression
from torchdyno.training.base import FitResult, Learner


def _reconstruction_data(T=300, B=4, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(T, B, 1)
    return [(x, x.clone())]  # target = current input


def _model(seed=0):
    torch.manual_seed(seed)
    core = ESNCore(input_size=1, layer_sizes=[100], input_scaling=0.9, rho=0.9)
    return SequenceModel(core, RegressionHead(core.state_size, 1))


def test_is_learner():
    assert isinstance(RidgeRegression(), Learner)


def test_exported_from_optim():
    assert hasattr(torchdyno.optim, "RidgeRegression")


def test_fit_learns_reconstruction():
    model = _model()
    train = _reconstruction_data()
    result = RidgeRegression(l2=1e-6).fit(model, train)
    assert isinstance(result, FitResult)
    x, y = train[0]
    mse = torch.mean((model(x) - y) ** 2).item()
    var = torch.var(y).item()
    assert mse < 0.5 * var


def test_validation_selects_l2():
    model = _model()
    train = _reconstruction_data(seed=0)
    val = _reconstruction_data(seed=1)

    def mse(target, pred):
        return torch.mean((target - pred) ** 2).item()

    learner = RidgeRegression(l2=[1e-8, 1e-6, 1e-2, 1.0], score_fn=mse, mode="min")
    result = learner.fit(model, train, val)
    assert result.best["l2"] in (1e-8, 1e-6, 1e-2, 1.0)
    assert "score" in result.best


def test_store_matrices():
    model = _model()
    result = RidgeRegression(l2=1e-6, store_matrices=True).fit(model, _reconstruction_data())
    assert "A" in result.extras and "B" in result.extras
