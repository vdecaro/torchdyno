import torch

from torchdyno import create_head, list_heads
from torchdyno.heads import DenseHead, RegressionHead


def test_dense_head_output_shape_is_per_timestep():
    head = DenseHead(input_size=8, num_classes=3)
    states = torch.randn(5, 4, 8)  # (T, B, H)
    out = head(states)
    assert out.shape == (5, 4, 3)  # (T, B, C) — no pooling


def test_dense_head_set_weight():
    head = DenseHead(input_size=8, num_classes=3)
    w = torch.randn(8, 3)
    head.set_weight(w)
    assert torch.allclose(head.weight, w)


def test_dense_head_gradients_flow():
    head = DenseHead(input_size=8, num_classes=3)
    states = torch.randn(5, 4, 8, requires_grad=True)
    head(states).sum().backward()
    assert states.grad is not None
    assert head.weight.grad is not None


def test_dense_head_is_a_head():
    assert isinstance(DenseHead(4, 2), RegressionHead)  # DRY: subclass


def test_dense_head_registered():
    assert "dense" in list_heads()
    head = create_head("dense", input_size=8, num_classes=3)
    assert head(torch.randn(5, 4, 8)).shape == (5, 4, 3)
