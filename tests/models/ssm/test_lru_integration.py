import torch

from torchdyno.heads import RegressionHead
from torchdyno.model import SequenceModel
from torchdyno.models.ssm.core import LRUCore
from torchdyno.nn.init import Ring
from torchdyno.optim.ridge_regression import fit_readout
from torchdyno.training.backprop import BackpropTrainer


def test_frozen_lru_ridge_reconstructs_input():
    torch.manual_seed(0)
    # r_max=0.9 keeps memory moderate (mirrors the ESN rho=0.9 reconstruction test).
    core = LRUCore(input_size=1, n_modes=64, trainable=False,
                   init=Ring(r_min=0.0, r_max=0.9))
    head = RegressionHead(core.state_size, 1)
    model = SequenceModel(core, head)

    T, B = 300, 4
    x = torch.randn(T, B, 1)
    y = x.clone()

    readout, _a, _b = fit_readout(
        [(x, y)], preprocess_fn=lambda batch: core(batch).states, l2=1e-6
    )
    head.set_weight(readout.T)

    pred = model(x)
    assert pred.shape == (T, B, 1)
    mse = torch.mean((pred - y) ** 2).item()
    var = torch.var(y).item()
    assert mse < 0.5 * var           # fitted readout beats a mean predictor


def test_trainable_lru_backprop_reduces_loss():
    torch.manual_seed(0)
    core = LRUCore(input_size=1, n_modes=32, trainable=True)
    head = RegressionHead(core.state_size, 1)
    model = SequenceModel(core, head)

    T, B = 40, 16
    x = torch.randn(T, B, 1)
    y = x.clone()

    trainer = BackpropTrainer(
        loss_fn=torch.nn.MSELoss(), optimizer="adam", lr=1e-2, epochs=40
    )
    result = trainer.fit(model, [(x, y)])
    losses = result.history["train_loss"]
    assert len(losses) == 40
    assert losses[-1] < losses[0]    # backprop through the complex scan learns
