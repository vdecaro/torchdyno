"""Sequence-classification head: pool over time, then linear."""

import math

import torch
from torch import (
    Tensor,
    nn,
)

from torchdyno.heads.base import Head
from torchdyno.registry import register_head

_POOLS = ("last", "mean")


@register_head("classification")
class ClassificationHead(Head):
    """Pool a state sequence over time, then apply a linear classifier.

    Args:
        input_size: the core state width ``H``.
        num_classes: number of output classes.
        pool: ``"last"`` uses the final timestep, ``"mean"`` averages over time.
        trainable: whether ``weight`` requires grad.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        pool: str = "last",
        trainable: bool = True,
    ):
        super().__init__()
        if pool not in _POOLS:
            raise ValueError(f"Unknown pool: {pool}. Choose from {_POOLS}.")
        self.pool = pool
        self.weight = nn.Parameter(
            torch.normal(
                mean=0.0,
                std=1.0 / math.sqrt(input_size),
                size=(input_size, num_classes),
            ),
            requires_grad=trainable,
        )

    def forward(self, states: Tensor) -> Tensor:
        if self.pool == "last":
            pooled = states[-1]
        else:
            pooled = states.mean(dim=0)
        return pooled @ self.weight

    def set_weight(self, weight: Tensor) -> None:
        self.weight.data = weight.to(self.weight)
