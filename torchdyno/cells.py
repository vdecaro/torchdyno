"""Cells: one step of a recurrence, ``cell(u_t, h) -> h'``.

The new state is also the cell's output, so a layer such as
:class:`~torchdyno.layers.Recurrent` only has to collect the states. Cells take
their matrices as arguments, so the same update rule works with any matrix
structure from :mod:`torchdyno.matrices`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from torchdyno.matrices import LinearMap


class Cell(nn.Module):
    """Base class for cells: ``forward(u_t, h) -> h'`` on ``(B, state_size)`` states."""

    state_size: int


class Gate(nn.Module):
    """A per-unit step size in ``(lo, hi)``, driven by the input and the state.

    ``lo + (hi - lo) * sigmoid(W u + b + a ⊙ h)``, with ``a`` starting at zero so
    the step size starts input-only.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bounds: tuple[float, float] = (1e-5, 0.2),
        dtype: torch.dtype = torch.float32,
    ):
        """Initializes the gate.

        Args:
            in_features: the input width.
            out_features: the state width.
            bounds: the ``(lo, hi)`` range of the step size.
            dtype: the data type.
        """
        super().__init__()
        self.lo, self.hi = bounds
        self.input = nn.Linear(in_features, out_features, dtype=dtype)
        self.state = nn.Parameter(torch.zeros(out_features, dtype=dtype))

    def forward(self, u: Tensor, h: Tensor) -> Tensor:
        """Compute the step size for one timestep.

        Args:
            u: the input, shape ``(B, in_features)``.
            h: the state, shape ``(B, out_features)``.

        Returns:
            The step size, shape ``(B, out_features)``.
        """
        logits = self.input(u) + self.state * h
        return self.lo + (self.hi - self.lo) * torch.sigmoid(logits)


class EulerCell(Cell):
    """One Euler step of a leaky ODE: ``v' = v + ε ⊙ (-v + W φ(v) + L v + B u)``.

    ``ε`` is a constant or a module such as :class:`Gate` that computes it from
    the input and the state. With a block-diagonal ``W`` and a skew-symmetric
    ``L`` this is an assembly of coupled modules.
    """

    def __init__(
        self,
        input: LinearMap,
        recurrent: LinearMap,
        coupling: LinearMap | None = None,
        step: float | nn.Module = 0.01,
        activation: Callable[[Tensor], Tensor] = torch.relu,
    ):
        """Initializes the cell.

        Args:
            input: the input matrix ``B``.
            recurrent: the matrix ``W`` applied to ``φ(v)``.
            coupling: an optional matrix ``L`` applied to ``v``.
            step: the step size ``ε``, or a module ``step(u, v) -> ε``.
            activation: the activation ``φ``.
        """
        super().__init__()
        self.input = input
        self.recurrent = recurrent
        self.coupling = coupling
        self.step = step
        self.activation = activation
        self.state_size = recurrent.out_features

    def forward(self, u: Tensor, v: Tensor) -> Tensor:
        """Advance the state by one timestep.

        Args:
            u: the input, shape ``(B, F)``.
            v: the state, shape ``(B, state_size)``.

        Returns:
            The new state.
        """
        drive = -v + self.recurrent(self.activation(v))
        if self.coupling is not None:
            drive = drive + self.coupling(v)
        drive = drive + self.input(u)
        step = self.step(u, v) if isinstance(self.step, nn.Module) else self.step
        return v + step * drive


class LeakyCell(Cell):
    """A leaky-integrator unit: ``h' = (1 - a) h + a φ(g ⊙ (B u + W h) + c)``.

    The gain ``g`` and bias ``c`` start at one and zero, an exact no-op, and
    change only under intrinsic plasticity or backprop.
    """

    def __init__(
        self,
        input: LinearMap,
        recurrent: LinearMap,
        leak: float = 1.0,
        activation: Callable[[Tensor], Tensor] = torch.tanh,
    ):
        """Initializes the cell.

        Args:
            input: the input matrix ``B``, with a bias if the units need one.
            recurrent: the recurrent matrix ``W``.
            leak: the leak rate ``a``.
            activation: the activation ``φ``.
        """
        super().__init__()
        self.input = input
        self.recurrent = recurrent
        self.leak = leak
        self.activation = activation
        self.state_size = recurrent.out_features
        dtype = next(recurrent.parameters()).dtype
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
