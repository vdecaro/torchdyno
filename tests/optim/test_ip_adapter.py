import pytest
import torch

from torchdyno.heads import RegressionHead
from torchdyno.model import SequenceModel
from torchdyno.models.esn.core import ESNCore
from torchdyno.models.esn.reservoir import Reservoir
from torchdyno.optim import IntrinsicPlasticity, RidgeRegression


def test_ip_requires_net_gain_and_bias():
    res = Reservoir(input_size=2, hidden_size=8, net_gain_and_bias=False)
    ip = IntrinsicPlasticity(learning_rate=0.01, mu=0.0, sigma=0.1)
    with pytest.raises(ValueError):
        ip.compile(res)


def test_ip_lifecycle_updates_and_detaches():
    torch.manual_seed(0)
    res = Reservoir(input_size=2, hidden_size=8, net_gain_and_bias=True)
    res.train()
    ip = IntrinsicPlasticity(learning_rate=0.01, mu=0.0, sigma=0.1)
    ip.compile(res)
    assert ip.compiled

    a0 = res.net_a.detach().clone()
    b0 = res.net_b.detach().clone()
    x = torch.randn(20, 4, 2)
    for _ in range(10):
        res(x)          # forward stashes intermediates (train mode)
        ip.backward()   # closed-form net_a/net_b grads
        ip.step()       # normalize + SGD

    assert not torch.allclose(res.net_a, a0)
    assert not torch.allclose(res.net_b, b0)
    assert torch.isfinite(res.net_a).all() and torch.isfinite(res.net_b).all()

    ip.detach()
    assert not ip.compiled
    out = res(x)  # default forward works again
    assert out.shape == (20, 4, 8) and torch.isfinite(out).all()


def test_ip_pretrain_then_ridge_readout():
    torch.manual_seed(0)
    core = ESNCore(
        input_size=1,
        layer_sizes=[64],
        net_gain_and_bias=True,
        input_scaling=0.9,
        rho=0.9,
    )
    model = SequenceModel(core, RegressionHead(core.state_size, 1))

    # IP-pretrain the (single) reservoir's gain/bias, then detach.
    res = core.reservoirs[0]
    res.train()
    ip = IntrinsicPlasticity(learning_rate=0.01, mu=0.0, sigma=0.1)
    ip.compile(res)
    x = torch.randn(200, 4, 1)
    for _ in range(5):
        res(x)
        ip.backward()
        ip.step()
    ip.detach()

    # Fit the readout in closed form through the Learner surface.
    RidgeRegression(l2=1e-6).fit(model, [(x, x.clone())])
    mse = torch.mean((model(x) - x) ** 2).item()
    assert mse < 0.5 * torch.var(x).item()
