import pytest
import torch

from torchdyno.objectives import (
    F1,
    MSE,
    NRMSE,
    Accuracy,
    BalancedAccuracy,
    ConfusionMatrix,
    MemoryCapacity,
    Regression,
    SequenceClassification,
    Task,
    TimestepClassification,
)


def test_regression_metrics():
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    pred = target + 1.0
    assert MSE()(pred, target) == pytest.approx(1.0)
    assert NRMSE()(pred, target) == pytest.approx(
        1.0 / target.var(unbiased=False).sqrt().item()
    )
    assert NRMSE()(target, target) == 0.0


def test_nrmse_normalises_each_output_by_its_own_variance():
    target = torch.stack([torch.randn(50), 1e-2 * torch.randn(50)], dim=-1)
    target = target - target.mean(0)
    pred = target.clone()
    pred[:, 1] = 0.0  # the small output is entirely wrong
    assert NRMSE()(pred, target) == pytest.approx(0.5, abs=1e-4)
    assert NRMSE()(pred[:, None], target[:, None]) == NRMSE()(pred, target)


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


@pytest.mark.parametrize("score", [MSE(), NRMSE(), Regression().loss])
def test_regression_scores_reject_targets_of_another_shape(score):
    target = torch.randn(8)
    with pytest.raises(ValueError, match="same shape"):
        score(target[:, None], target)


def test_accuracy_rejects_predictions_without_a_class_dimension():
    with pytest.raises(ValueError, match="class dimension"):
        Accuracy()(torch.randn(8, 3), torch.zeros(8, 1, dtype=torch.long))


def test_rejects_unknown_primary():
    with pytest.raises(ValueError, match="Unknown primary"):
        Regression(primary="mae")


def _logits(labels, classes=3):
    return torch.nn.functional.one_hot(torch.tensor(labels), classes).float()


def test_class_balanced_metrics_from_the_confusion_matrix():
    target = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
    pred = _logits([0, 0, 0, 1, 1, 0, 2, 0])
    confusion = ConfusionMatrix()(pred, target)
    assert confusion.tolist() == [[3, 1, 0], [1, 1, 0], [1, 0, 1]]
    assert Accuracy()(pred, target) == pytest.approx(5 / 8)
    assert BalancedAccuracy()(pred, target) == pytest.approx(
        (3 / 4 + 1 / 2 + 1 / 2) / 3
    )
    f1 = [2 * 3 / (2 * 3 + 2 + 1), 2 * 1 / (2 * 1 + 1 + 1), 2 * 1 / (2 * 1 + 0 + 1)]
    assert F1()(pred, target) == pytest.approx(sum(f1) / 3)
    constant = _logits([0] * 8)
    assert BalancedAccuracy()(constant, target) == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="class dimension"):
        F1()(pred, target[:, None])


def test_memory_capacity_sums_squared_correlations():
    torch.manual_seed(0)
    target = torch.randn(200, 3)
    pred = torch.stack([2 * target[:, 0] + 1, -target[:, 1], torch.zeros(200)], -1)
    assert MemoryCapacity()(pred, target) == pytest.approx(2.0)
    assert MemoryCapacity.mode == "max"


def test_tasks_take_their_loss_and_metrics_without_subclassing():
    task = Task(
        {"f1": F1(), "accuracy": Accuracy()},
        "f1",
        loss=torch.nn.functional.cross_entropy,
    )
    pred, target = _logits([0, 1, 2, 1]), torch.tensor([0, 1, 2, 2])
    assert task.primary is task._metrics["f1"]
    assert task.loss(pred, target) == torch.nn.functional.cross_entropy(pred, target)
    with pytest.raises(NotImplementedError, match="loss="):
        Task({"f1": F1()}, "f1").loss(pred, target)


def test_timestep_classification_scores_every_step():
    task = TimestepClassification()
    pred, target = torch.randn(5, 2, 3), torch.randint(3, (5, 2))
    flat = torch.nn.functional.cross_entropy(pred.flatten(0, 1), target.flatten())
    assert task.loss(pred, target) == flat
    assert task.metrics(pred, target)["accuracy"] == Accuracy()(
        pred.flatten(0, 1), target.flatten()
    )
