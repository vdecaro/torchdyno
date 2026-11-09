import pytest
import torch

from torchdyno.metrics import (
    MSE,
    NRMSE,
    Accuracy,
)
from torchdyno.tasks import (
    Regression,
    SequenceClassification,
)


def test_regression_metrics():
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    pred = target + 1.0
    assert MSE()(pred, target) == pytest.approx(1.0)
    assert NRMSE()(pred, target) == pytest.approx(
        1.0 / target.var(unbiased=False).sqrt().item()
    )
    assert NRMSE()(target, target) == 0.0


def test_accuracy():
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [3.0, 1.0]])
    assert Accuracy()(logits, torch.tensor([0, 1, 1])) == pytest.approx(2 / 3)
    assert Accuracy.mode == "max"


def test_regression_task():
    task = Regression(primary="mse")
    pred, target = torch.zeros(5, 2, 1), torch.ones(5, 2, 1)
    assert task.loss(pred, target).item() == pytest.approx(1.0)
    assert set(task.metrics(pred, target)) == {"nrmse", "mse"}
    assert isinstance(task.primary, MSE)


def test_classification_task():
    task = SequenceClassification()
    logits, labels = torch.tensor([[5.0, -5.0]]), torch.tensor([0])
    assert task.loss(logits, labels).item() < 1e-3
    assert task.metrics(logits, labels) == {"accuracy": 1.0}
    assert task.primary.mode == "max"


def test_rejects_unknown_primary():
    with pytest.raises(ValueError, match="Unknown primary"):
        Regression(primary="mae")
