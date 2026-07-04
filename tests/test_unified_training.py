import torch

from torchdyno.heads import ClassificationHead, RegressionHead
from torchdyno.model import SequenceModel
from torchdyno.models.assembly.adadiag import AdaDiagCore
from torchdyno.models.esn.core import ESNCore
from torchdyno.optim.ridge import RidgeRegression
from torchdyno.training.base import FitResult, Learner
from torchdyno.training.backprop import BackpropTrainer


def _fit_and_check(learner: Learner, model, train) -> FitResult:
    # Any Learner is driven identically through the same surface.
    result = learner.fit(model, train)
    assert isinstance(result, FitResult)
    return result


def test_rc_and_backprop_share_one_surface():
    torch.manual_seed(0)

    # Frozen-core RC model, trained in closed form.
    esn = SequenceModel(
        ESNCore(input_size=1, layer_sizes=[100], input_scaling=0.9, rho=0.9),
        RegressionHead(100, 1),
    )
    x_rc = torch.randn(300, 4, 1)
    rc_train = [(x_rc, x_rc.clone())]
    _fit_and_check(RidgeRegression(l2=1e-6), esn, rc_train)
    rc_mse = torch.mean((esn(x_rc) - x_rc) ** 2).item()
    assert rc_mse < 0.5 * torch.var(x_rc).item()

    # Differentiable assembly model, trained by backprop.
    torch.manual_seed(1)
    ada_core = AdaDiagCore(input_size=1, block_sizes=[16, 16], coupling_topology=[(0, 1)])
    ada = SequenceModel(ada_core, ClassificationHead(ada_core.state_size, 2, pool="last"))
    x_bp = torch.randn(30, 16, 1)
    y_bp = (x_bp.mean(dim=0).squeeze(-1) > 0).long()
    bp_train = [(x_bp, y_bp)]
    bp_result = _fit_and_check(
        BackpropTrainer(loss_fn=torch.nn.CrossEntropyLoss(), lr=1e-2, epochs=60),
        ada,
        bp_train,
    )
    assert bp_result.history["train_loss"][-1] < bp_result.history["train_loss"][0]
    acc = (ada(x_bp).argmax(dim=-1) == y_bp).float().mean().item()
    assert acc > 0.5  # better than chance on the (learnable) task
