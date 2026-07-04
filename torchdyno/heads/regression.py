"""Linear regression head."""

import math

import torch
from torch import (
    Tensor,
    nn,
)

from torchdyno.heads.base import Head
from torchdyno.registry import register_head


@register_head("regression")
class RegressionHead(Head):
    """A per-timestep linear readout: ``states @ weight``.

    Args:
        input_size: the core state width ``H``.
        output_size: the number of regression targets.
        trainable: whether ``weight`` requires grad (for backprop). Closed-form
            fitting via ``set_weight`` works regardless.
    """

    def __init__(self, input_size: int, output_size: int, trainable: bool = True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.normal(
                mean=0.0,
                std=1.0 / math.sqrt(input_size),
                size=(input_size, output_size),
            ),
            requires_grad=trainable,
        )

    def forward(self, states: Tensor) -> Tensor:
        return states @ self.weight

    def set_weight(self, weight: Tensor) -> None:
        self.weight.data = weight.to(self.weight)
