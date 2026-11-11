"""What every cell is, and the step-like values cells take."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

    from torch import Tensor

    # a constant, a per-unit tensor, or computed from the input and the state
    Step: TypeAlias = float | Tensor | Callable[[Tensor, Tensor], Tensor]


class Cell(nn.Module):
    """Base class for cells: ``forward(u_t, h) -> h'`` on ``(B, state_size)`` states."""

    state_size: int


def _width(name: str, actual: int, expected: int) -> None:
    """Raise unless a part of a cell has the width the state needs."""
    if actual != expected:
        raise ValueError(f"{name} has width {actual}, but the state needs {expected}.")


def _keep(module: nn.Module, name: str, value: Step) -> None:
    """Keep a step-like ``value`` on ``module``: a tensor as a buffer.

    A constant or a tensor must be positive; a callable ``value(u, h)`` is
    computed at every timestep.
    """
    if not callable(value) and not torch.all(torch.as_tensor(value) > 0):
        raise ValueError(f"{name} must be > 0, got {value}.")
    if isinstance(value, torch.Tensor):
        module.register_buffer(name, value)
    else:
        setattr(module, name, value)


def _at(value: Step, u: Tensor, h: Tensor) -> float | Tensor:
    """A step-like value at the current timestep."""
    return value(u, h) if callable(value) else value
