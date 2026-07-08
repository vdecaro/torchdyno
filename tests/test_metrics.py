import pytest
import torch

from torchdyno.metrics import MSE, MAE, NRMSE, Accuracy, MacroF1, Metric, to_indices


def test_regression_metrics_zero_for_perfect_prediction():
    x = torch.randn(5, 3, 2)
    assert MSE()(x, x.clone()) == pytest.approx(0.0)
    assert MAE()(x, x.clone()) == pytest.approx(0.0)
    assert NRMSE()(x, x.clone()) == pytest.approx(0.0)


def test_regression_metric_modes_and_names():
    assert (MSE().mode, MSE().name) == ("min", "mse")
    assert (MAE().mode, MAE().name) == ("min", "mae")
    assert (NRMSE().mode, NRMSE().name) == ("min", "nrmse")


def test_mse_and_mae_known_values():
    pred = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[0.0, 0.0]])
    assert MSE()(pred, target) == pytest.approx((1.0 + 4.0) / 2)
    assert MAE()(pred, target) == pytest.approx((1.0 + 2.0) / 2)


def test_nrmse_normalizes_by_target_std():
    # target var = 1.0 (values -1, 1 with population var 1); rmse = 1.0 -> nrmse = 1.0
    target = torch.tensor([-1.0, 1.0])
    pred = torch.tensor([0.0, 2.0])  # errors of 1.0 each -> rmse 1.0
    assert NRMSE()(pred, target) == pytest.approx(1.0, abs=1e-5)


def test_accuracy_known_values_and_mode():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])  # preds: 0,1,0
    target = torch.tensor([0, 1, 1])  # 2/3 correct
    assert Accuracy()(logits, target) == pytest.approx(2 / 3)
    assert (Accuracy().mode, Accuracy().name) == ("max", "accuracy")


def test_accuracy_accepts_one_hot_targets():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])  # preds 0, 1
    onehot = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # classes 0, 1 -> all correct
    assert Accuracy()(logits, onehot) == pytest.approx(1.0)


def test_accuracy_dense_shape():
    # (T, B, C) logits with (T, B) targets
    logits = torch.zeros(4, 2, 3)
    logits[..., 1] = 1.0  # every prediction is class 1
    target = torch.ones(4, 2, dtype=torch.long)  # all class 1 -> perfect
    assert Accuracy()(logits, target) == pytest.approx(1.0)


def test_macro_f1_perfect_and_known():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])  # 0,1,0,1
    target = torch.tensor([0, 1, 0, 1])
    assert MacroF1()(logits, target) == pytest.approx(1.0)
    assert (MacroF1().mode, MacroF1().name) == ("max", "macro_f1")


def test_macro_f1_nontrivial_case():
    # argmax(logits) = [0, 0, 1, 2, 2]; target = [0, 1, 1, 2, 0]
    # per-class F1: class0=1/2, class1=2/3, class2=2/3 -> macro = 11/18
    logits = torch.tensor([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ])
    target = torch.tensor([0, 1, 1, 2, 0])
    assert MacroF1()(logits, target) == pytest.approx(11 / 18)


def test_to_indices_argmaxes_one_hot_only():
    pred = torch.zeros(2, 3)
    onehot = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert torch.equal(to_indices(pred, onehot), torch.tensor([1, 0]))
    idx = torch.tensor([1, 0])
    assert torch.equal(to_indices(pred, idx), torch.tensor([1, 0]))


def test_metric_base_is_abstract_callable():
    m = Metric()
    with pytest.raises(NotImplementedError):
        m(torch.zeros(1), torch.zeros(1))
