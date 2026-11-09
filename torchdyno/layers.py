"""Layers: modules that map a ``(T, B, F)`` sequence to a ``(T, B, H)`` one."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor

    from torchdyno.cells import Cell


class Recurrent(nn.Module):
    """Run a cell over time, one step after the other."""

    def __init__(self, cell: Cell):
        """Initializes the layer.

        Args:
            cell: the cell to run.
        """
        super().__init__()
        self.cell = cell

    def forward(self, x: Tensor, h0: Tensor | None = None) -> Tensor:
        """Run the cell over ``x``.

        Args:
            x: the input sequence, shape ``(T, B, F)``.
            h0: the state to start from, e.g. the last output of a previous
                call; ``None`` starts from zeros.

        Returns:
            The state after every timestep, shape ``(T, B, state_size)``.
        """
        h = x.new_zeros(x.shape[1], self.cell.state_size) if h0 is None else h0
        states = []
        for u in x:
            h = self.cell(u, h)
            states.append(h)
        return torch.stack(states)
