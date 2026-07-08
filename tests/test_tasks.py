import pytest
import torch
from torch import nn

from torchdyno import SequenceModel
from torchdyno.heads import RegressionHead
from torchdyno.models.assembly import SCNCore
from torchdyno.models.esn import ESNCore
from torchdyno.optim import RidgeRegression
from torchdyno.tasks import DenseLabeling, Regression, SequenceClassification, Task
from torchdyno.training import BackpropTrainer
from torchdyno.training.base import FitResult


def test_regression_task_loss_and_metrics():
    task = Regression(primary="nrmse")
    x = torch.randn(6, 3, 2)
    assert task.loss(x, x.clone()).item() == pytest.approx(0.0)
    m = task.metrics(x, x.clone())
    assert set(m) == {"nrmse", "mse", "mae"}
    assert task.primary.name == "nrmse" and task.primary.mode == "min"


def test_regression_primary_selectable():
    assert Regression(primary="mae").primary.name == "mae"
    with pytest.raises(ValueError):
        Regression(primary="nope")


def test_sequence_classification_loss_and_metrics():
    task = SequenceClassification(num_classes=3)
    logits = torch.randn(4, 3)  # (B, C)
    target = torch.tensor([0, 1, 2, 1])
    assert task.loss(logits, target).item() > 0
    assert set(task.metrics(logits, target)) == {"accuracy", "macro_f1"}
    assert task.primary.name == "accuracy" and task.primary.mode == "max"


def test_sequence_classification_loss_accepts_one_hot():
    task = SequenceClassification(num_classes=2)
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    onehot = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    idx = torch.tensor([0, 1])
    assert task.loss(logits, onehot).item() == pytest.approx(task.loss(logits, idx).item())


def test_dense_labeling_loss_over_time():
    task = DenseLabeling(num_classes=3)
    logits = torch.randn(5, 4, 3)  # (T, B, C)
    target = torch.randint(0, 3, (5, 4))  # (T, B)
    assert task.loss(logits, target).item() > 0
    assert set(task.metrics(logits, target)) == {"accuracy", "macro_f1"}


def test_tasks_are_task_instances():
    assert isinstance(Regression(), Task)
    assert isinstance(SequenceClassification(2), Task)
    assert isinstance(DenseLabeling(2), Task)


def test_integration_ridge_with_regression_task_beats_mean():
    torch.manual_seed(0)
    core = ESNCore(input_size=1, layer_sizes=[100], input_scaling=0.9, rho=0.9)
    model = SequenceModel(core, RegressionHead(core.state_size, 1))
    x = torch.randn(300, 4, 1)
    task = Regression(primary="nrmse")
    RidgeRegression(l2=1e-6, score_fn=task.primary, mode=task.primary.mode).fit(
        model, [(x, x.clone())]
    )
    mse = torch.mean((model(x) - x) ** 2).item()
    assert mse < 0.5 * torch.var(x).item()


def test_integration_backprop_with_task_returns_fitresult():
    torch.manual_seed(0)
    core = SCNCore(input_size=1, block_sizes=[8, 8], coupling_topology="ring")
    model = SequenceModel(core, RegressionHead(core.state_size, 1))
    x = torch.randn(20, 4, 1)
    task = Regression()
    result = BackpropTrainer(loss_fn=task.loss, score_fn=task.primary, epochs=2).fit(
        model, [(x, x.clone())]
    )
    assert isinstance(result, FitResult)
