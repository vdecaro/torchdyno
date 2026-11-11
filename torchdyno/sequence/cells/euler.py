"""EulerCell and the parts it integrates: LeakyField and the Gate step."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from torchdyno.matrices.ops import LinearMap
from torchdyno.sequence.cells.base import (
    Cell,
    _at,
    _keep,
    _width,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from torchdyno.matrices import LinearMap
    from torchdyno.sequence.cells.base import Step


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
        dtype: torch.dtype | None = None,
    ):
        """Initializes the gate.

        Args:
            in_features: the input width.
            out_features: the state width.
            bounds: the ``(lo, hi)`` range of the step size.
            dtype: the data type; ``None`` takes torch's default.

        Raises:
            ValueError: unless ``0 <= lo < hi``.
        """
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        if not 0 <= bounds[0] < bounds[1]:
            raise ValueError(f"Need 0 <= lo < hi, got {bounds}.")
        self.in_features, self.out_features = in_features, out_features
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


class LeakyField(nn.Module):
    """A leaky ODE's vector field: ``f(u, v) = -v + W φ(v) + L v + B u``.

    With a block-diagonal ``W`` and a skew-symmetric ``L``, an assembly of
    coupled modules; integrate it with :class:`EulerCell`.
    """

    def __init__(
        self,
        input: LinearMap,
        recurrent: LinearMap,
        coupling: LinearMap | None = None,
        activation: Callable[[Tensor], Tensor] = torch.relu,
    ):
        """Initializes the field.

        Args:
            input: the input matrix ``B``.
            recurrent: the matrix ``W`` applied to ``φ(v)``.
            coupling: an optional matrix ``L`` applied to ``v``.
            activation: the activation ``φ``.

        Raises:
            ValueError: if a matrix does not match the state.
        """
        super().__init__()
        size = recurrent.out_features
        _width("recurrent.in_features", recurrent.in_features, size)
        _width("input.out_features", input.out_features, size)
        if coupling is not None:
            _width("coupling.in_features", coupling.in_features, size)
            _width("coupling.out_features", coupling.out_features, size)
        self.input = input
        self.recurrent = recurrent
        self.coupling = coupling
        self.activation = activation
        self.state_size = size

    def forward(self, u: Tensor, v: Tensor) -> Tensor:
        """The time derivative of the state.

        Args:
            u: the input, shape ``(B, F)``.
            v: the state, shape ``(B, state_size)``.

        Returns:
            ``f(u, v)``, shape ``(B, state_size)``.
        """
        drive = -v + self.recurrent(self.activation(v))
        if self.coupling is not None:
            drive = drive + self.coupling(v)
        return drive + self.input(u)


class EulerCell(Cell):
    """One explicit Euler step of an ODE: ``h' = h + ε ⊙ f(u, h)``.

    ``f`` is any vector field, e.g. :class:`LeakyField` for an assembly of
    coupled modules, or a module of one's own. ``ε`` is a constant, a per-unit
    tensor, or computed from the input and the state, e.g. by a :class:`Gate`.
    """

    step: Step

    def __init__(
        self,
        field: Callable[[Tensor, Tensor], Tensor],
        step: Step = 0.01,
        state_size: int | None = None,
    ):
        """Initializes the cell.

        Args:
            field: the vector field ``f(u, h)``.
            step: the step size ``ε``: a constant, a tensor kept as a buffer, or
                a callable ``step(u, h) -> ε``.
            state_size: the state width; ``None`` takes ``field.state_size``.

        Raises:
            ValueError: if the state width is unknown, the step does not match
                it, or a constant step is not positive.
        """
        super().__init__()
        size = getattr(field, "state_size", None) if state_size is None else state_size
        if not isinstance(size, int):
            raise ValueError("Give state_size= for a field without a state_size.")
        step_width = getattr(step, "out_features", None)
        if isinstance(step_width, int):
            _width("step.out_features", step_width, size)
        if isinstance(step, torch.Tensor) and step.numel() != 1:
            _width("step", step.shape[-1], size)
        self.field = field
        _keep(self, "step", step)
        self.state_size = size

    def forward(self, u: Tensor, h: Tensor) -> Tensor:
        """Advance the state by one timestep.

        Args:
            u: the input, shape ``(B, F)``.
            h: the state, shape ``(B, state_size)``.

        Returns:
            The new state.
        """
        return h + _at(self.step, u, h) * self.field(u, h)
