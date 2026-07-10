import torch

from torchdyno.heads import RegressionHead
from torchdyno.model import SequenceModel
from torchdyno.models.ssm.s4d import S4DCore
from torchdyno.nn.init import S4DLin
from torchdyno.optim.ridge_regression import fit_readout
from torchdyno.training.backprop import BackpropTrainer


def test_frozen_s4d_ridge_reconstructs_input():
    torch.manual_seed(0)
    # Larger dt ⇒ shorter memory (|Ā| = exp(dt·Re A)) so the instantaneous input is
    # cleanly recoverable (mirrors the ESN rho=0.9 / LRU r_max=0.9 reconstruction test).
    core = S4DCore(
        input_size=1, n_modes=64, trainable=False, init=S4DLin(dt_min=0.05, dt_max=0.5)
    )
    head = RegressionHead(core.state_size, 1)
    model = SequenceModel(core, head)

    T, B = 300, 4
    x = torch.randn(T, B, 1)
    y = x.clone()

    readout, _a, _b = fit_readout(
        [(x, y)], preprocess_fn=lambda b: core(b).states, l2=1e-6
    )
    head.set_weight(readout.T)

    pred = model(x)
    assert pred.shape == (T, B, 1)
    mse = torch.mean((pred - y) ** 2).item()
    var = torch.var(y).item()
    assert mse < 0.5 * var


def test_trainable_s4d_backprop_reduces_loss():
    torch.manual_seed(0)
    core = S4DCore(input_size=1, n_modes=32, trainable=True)
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
    assert losses[-1] < losses[0]
