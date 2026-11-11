"""LeakyCell: a leaky-integrator reservoir unit."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from torchdyno.matrices.ops import LinearMap
from torchdyno.sequence.cells.base import (
    Cell,
    _width,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from torchdyno.matrices import LinearMap


class LeakyCell(Cell):
    """A leaky-integrator unit: ``h' = (1 - a) h + a φ(g ⊙ (B u + W h) + c)``.

    The gain ``g`` and bias ``c`` start at one and zero, an exact no-op, and
    change only under intrinsic plasticity or backprop.
    """

    leak: float | Tensor

    def __init__(
        self,
        input: LinearMap,
        recurrent: LinearMap,
        leak: float | Tensor = 1.0,
        activation: Callable[[Tensor], Tensor] = torch.tanh,
    ):
        """Initializes the cell.

        Args:
            input: the input matrix ``B``, with a bias if the units need one.
            recurrent: the recurrent matrix ``W``.
            leak: the leak rate ``a``: a constant, or a per-unit tensor kept as
                a buffer.
            activation: the activation ``φ``.

        Raises:
            ValueError: if a matrix or the leak does not match the state, or the
                leak is outside ``(0, 1]``.
        """
        super().__init__()
        rate = torch.as_tensor(leak)
        if not torch.all((rate > 0) & (rate <= 1)):
            raise ValueError(f"leak must be in (0, 1], got {leak}.")
        size = recurrent.out_features
        _width("recurrent.in_features", recurrent.in_features, size)
        _width("input.out_features", input.out_features, size)
        if isinstance(leak, torch.Tensor) and leak.numel() != 1:
            _width("leak", leak.shape[-1], size)
        self.input = input
        self.recurrent = recurrent
        if isinstance(leak, torch.Tensor):
            self.register_buffer("leak", leak)
        else:
            self.leak = leak
        self.activation = activation
        self.state_size = recurrent.out_features
        weight = next(recurrent.parameters(), None)
        dtype = torch.get_default_dtype() if weight is None else weight.dtype
        self.gain = nn.Parameter(torch.ones(self.state_size, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(self.state_size, dtype=dtype))

    def preactivation(self, u: Tensor, h: Tensor) -> Tensor:
        """The input to the gain: ``B u + W h``.

        Args:
            u: the input, shape ``(B, F)``.
            h: the state, shape ``(B, state_size)``.

        Returns:
            The preactivation, shape ``(B, state_size)``.
        """
        return self.input(u) + self.recurrent(h)

    def forward(self, u: Tensor, h: Tensor) -> Tensor:
        """Advance the state by one timestep.

        Args:
            u: the input, shape ``(B, F)``.
            h: the state, shape ``(B, state_size)``.

        Returns:
            The new state.
        """
        drive = self.activation(self.gain * self.preactivation(u, h) + self.bias)
        return (1 - self.leak) * h + self.leak * drive
