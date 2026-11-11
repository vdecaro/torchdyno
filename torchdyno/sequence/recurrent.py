"""Running a cell over time: step by step, or by a parallel scan."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

import torch
from torch import nn

from torchdyno.sequence._checks import (
    _absent,
    _check,
)
from torchdyno.sequence.cells.base import Cell
from torchdyno.sequence.cells.linear import LinearDiagonalCell

if TYPE_CHECKING:
    from torch import Tensor

    from torchdyno.sequence import (
        Cell,
        LinearDiagonalCell,
    )


class Recurrent(nn.Module):
    """Run a cell over time, one step after the other.

    With ``carry=True`` the layer keeps its last state, detached, and the next
    call starts from it, e.g. for truncated backprop through time or a stream
    fed in chunks; :meth:`reset_state` starts over.

    To speed the loop up, compile the cell in place, ``layer.cell.compile()``,
    rather than the model: compiling the model unrolls the loop over every
    timestep, which takes far longer to compile.
    """

    def __init__(self, cell: Cell, carry: bool = False):
        """Initializes the layer.

        Args:
            cell: the cell to run.
            carry: start every call from the last state of the previous one.
        """
        super().__init__()
        self.cell = cell
        self.carry = carry
        self.state: Tensor | None = None

    def reset_state(self) -> None:
        """Forget the carried state: the next call starts from zeros."""
        self.state = None

    def _start(self, x: Tensor, h0: Tensor | None) -> Tensor | None:
        """Check the input and pick the state to start from."""
        if h0 is None and self.carry:
            h0 = self.state
        _check(x, h0, self.cell.state_size)
        return h0

    def _finish(self, states: Tensor, absent: Tensor) -> Tensor:
        """Keep the last state when carrying; NaN where a step is absent."""
        if self.carry:
            self.state = states[-1].detach()
        return states.masked_fill(absent[..., None], torch.nan)

    def forward(self, x: Tensor, h0: Tensor | None = None) -> Tensor:
        """Run the cell over ``x``.

        Args:
            x: the input sequence, shape ``(T, B, F)``.
            h0: the state to start from, e.g. the last output of a previous
                call; ``None`` starts from zeros, or from the carried state.

        Returns:
            The state after every timestep, shape ``(T, B, state_size)``; at a
            step a sequence does not have, the state holds and the output is
            NaN.

        Raises:
            ValueError: if ``x`` or the starting state has the wrong shape.
        """
        h0 = self._start(x, h0)
        absent = _absent(x)
        x = x.masked_fill(absent[..., None], 0)
        h = x.new_zeros(x.shape[1], self.cell.state_size) if h0 is None else h0
        states = []
        for u, gap in zip(x, absent, strict=True):
            step = self.cell(u, h)
            h = torch.where(gap[:, None], h, step) if gap.any() else step
            states.append(h)
        return self._finish(torch.stack(states), absent)


class Scan(Recurrent):
    """Run a :class:`~torchdyno.sequence.LinearDiagonalCell` with a parallel scan.

    Same contract as :class:`Recurrent`, but ``ceil(log2 T)`` steps over the whole
    sequence replace the ``T`` sequential ones, which the linear recurrence
    allows.
    """

    cell: LinearDiagonalCell

    def __init__(self, cell: LinearDiagonalCell, carry: bool = False):
        """Initializes the layer.

        Args:
            cell: the linear cell to run.
            carry: start every call from the last state of the previous one.

        Raises:
            TypeError: if the cell has no ``coefficients(u)`` to scan.
        """
        if not hasattr(cell, "coefficients"):
            raise TypeError(
                f"Scan needs a linear cell with coefficients(u), such as "
                f"LinearDiagonalCell, not {type(cell).__name__}; run it with Recurrent."
            )
        super().__init__(cell, carry)

    def forward(self, x: Tensor, h0: Tensor | None = None) -> Tensor:
        """Run the cell over ``x``.

        Args:
            x: the input sequence, shape ``(T, B, F)``.
            h0: the state to start from, e.g. the last output of a previous
                call; ``None`` starts from zeros, or from the carried state.

        Returns:
            The state after every timestep, shape ``(T, B, state_size)``.

        Raises:
            ValueError: if ``x`` or the starting state has the wrong shape.
        """
        h0 = self._start(x, h0)
        absent = _absent(x)
        a, b = self.cell.coefficients(x.masked_fill(absent[..., None], 0))
        a, b = torch.broadcast_tensors(a, b)
        if absent.any():  # hold the state: h_t = 1 h_(t-1) + 0
            gap = absent[..., None]
            a, b = a.masked_fill(gap, 1), b.masked_fill(gap, 0)
        if h0 is not None:
            first = b[0] + a[0] * self.cell.unpack(h0)
            b = torch.cat([first.unsqueeze(0), b[1:]])
        return self._finish(self.cell.pack(_scan(a, b)), absent)


def _scan(a: Tensor, b: Tensor) -> Tensor:
    """Hillis-Steele scan of ``h_t = a_t h_(t-1) + b_t`` over dim 0, from zero."""
    shift = 1
    while shift < b.shape[0]:
        a, b = (
            torch.cat([a[:shift], a[shift:] * a[:-shift]]),
            torch.cat([b[:shift], a[shift:] * b[:-shift] + b[shift:]]),
        )
        shift *= 2
    return b
