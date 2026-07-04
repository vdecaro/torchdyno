import pytest
import torch

from torchdyno.heads import ClassificationHead, Head


def test_is_head():
    assert isinstance(ClassificationHead(8, 3), Head)


def test_last_pool_shape():
    head = ClassificationHead(8, 3, pool="last")
    out = head(torch.randn(10, 4, 8))  # (T, B, H)
    assert out.shape == (4, 3)  # (B, num_classes)


def test_mean_pool_shape():
    head = ClassificationHead(8, 5, pool="mean")
    out = head(torch.randn(10, 4, 8))
    assert out.shape == (4, 5)


def test_mean_pool_uses_time_mean():
    torch.manual_seed(0)
    head = ClassificationHead(8, 2, pool="mean")
    states = torch.randn(10, 4, 8)
    expected = states.mean(dim=0) @ head.weight
    assert torch.allclose(head(states), expected)


def test_rejects_bad_pool():
    with pytest.raises(ValueError):
        ClassificationHead(8, 3, pool="max")


def test_set_weight():
    head = ClassificationHead(8, 3)
    w = torch.ones(8, 3)
    head.set_weight(w)
    assert torch.equal(head.weight.data, w)
