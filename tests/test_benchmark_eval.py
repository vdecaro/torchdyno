import torch

from torchdyno.benchmark.runner import _evaluate
from torchdyno.tasks import DenseLabeling, Regression, SequenceClassification


def _identity(x):
    return x


def test_evaluate_multibatch_dense_concats_along_batch_dim():
    torch.manual_seed(0)
    T, C = 6, 3
    logits = [torch.randn(T, 2, C), torch.randn(T, 3, C)]      # (T, B_i, C)
    targets = [torch.randint(0, C, (T, 2)), torch.randint(0, C, (T, 3))]  # (T, B_i)
    got = _evaluate(_identity, list(zip(logits, targets)), DenseLabeling(num_classes=C))
    want = DenseLabeling(num_classes=C).metrics(
        torch.cat(logits, dim=1), torch.cat(targets, dim=1)
    )
    assert got == want


def test_evaluate_multibatch_regression_concats_along_batch_dim():
    torch.manual_seed(0)
    T = 4
    preds = [torch.randn(T, 2, 1), torch.randn(T, 3, 1)]
    targets = [torch.randn(T, 2, 1), torch.randn(T, 3, 1)]
    got = _evaluate(_identity, list(zip(preds, targets)), Regression("nrmse"))
    want = Regression("nrmse").metrics(torch.cat(preds, dim=1), torch.cat(targets, dim=1))
    assert got == want


def test_evaluate_multibatch_seqclass_concats_along_dim0():
    torch.manual_seed(0)
    logits = [torch.randn(2, 3), torch.randn(4, 3)]            # (B_i, C)
    targets = [torch.randint(0, 3, (2,)), torch.randint(0, 3, (4,))]
    got = _evaluate(_identity, list(zip(logits, targets)), SequenceClassification(3))
    want = SequenceClassification(3).metrics(
        torch.cat(logits, dim=0), torch.cat(targets, dim=0)
    )
    assert got == want
