import torch

from torchdyno.heads import Head, RegressionHead


def test_regression_head_is_head():
    assert isinstance(RegressionHead(8, 2), Head)


def test_weight_shape():
    head = RegressionHead(8, 3)
    assert head.weight.shape == (8, 3)


def test_forward_time_first_shape():
    head = RegressionHead(8, 3)
    states = torch.randn(10, 4, 8)  # (T, B, H)
    out = head(states)
    assert out.shape == (10, 4, 3)


def test_forward_flat_shape():
    head = RegressionHead(8, 3)
    out = head(torch.randn(20, 8))
    assert out.shape == (20, 3)


def test_set_weight():
    head = RegressionHead(8, 3)
    w = torch.ones(8, 3)
    head.set_weight(w)
    assert torch.equal(head.weight.data, w)


def test_trainable_flag():
    assert RegressionHead(8, 3, trainable=True).weight.requires_grad
    assert not RegressionHead(8, 3, trainable=False).weight.requires_grad
