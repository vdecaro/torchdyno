"""FeedbackCell: a cell that also reads a readout of its own state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torchdyno.sequence.cells.base import Cell

if TYPE_CHECKING:
    from torch import (
        Tensor,
        nn,
    )


class FeedbackCell(Cell):
    """Close a loop: ``h' = cell([u; z], h)`` with ``z = readout(h)``.

    The readout's output at the previous state is fed back as extra input, so
    the inner cell's input matrix takes ``F + O`` features. Share the readout
    with the model's output, ``nn.Sequential(Recurrent(FeedbackCell(cell,
    readout)), readout)``, to feed the model's own prediction back; run it on a
    ``(T, B, 0)`` input to generate with no input at all. Train the readout in
    the loop with [`FORCE`][torchdyno.learners.FORCE].
    """

    def __init__(self, cell: Cell, readout: nn.Module):
        """Initializes the cell.

        Args:
            cell: the cell to run, reading ``[u; z]``.
            readout: the map from the state to the fed-back ``z``.
        """
        super().__init__()
        self.cell = cell
        self.readout = readout
        self.state_size = cell.state_size

    def forward(self, u: Tensor, h: Tensor) -> Tensor:
        """Advance the state by one timestep.

        Args:
            u: the input, shape ``(B, F)``.
            h: the state, shape ``(B, state_size)``.

        Returns:
            The inner cell's new state.
        """
        return self.cell(torch.cat([u, self.readout(h)], dim=-1), h)
