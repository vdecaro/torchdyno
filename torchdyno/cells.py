"""Cells: one step of a recurrence, ``cell(u_t, h) -> h'``.

The new state is also the cell's output, so a layer such as
:class:`~torchdyno.layers.Recurrent` only has to collect the states. Cells take
their matrices as arguments, so the same update rule works with any matrix
structure from :mod:`torchdyno.matrices`.
"""

from __future__ import annotations

import math
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


class RingDiagonal(nn.Module):
    """LRU's stable complex diagonal ``λ = exp(-e^nu + iθ)``, with ``sqrt(1 - |λ|²)``.

    From Orvieto et al., "Resurrecting Recurrent Neural Networks for Long
    Sequences" (2023). ``|λ| < 1`` for every ``nu``, so training cannot make the
    recurrence unstable. ``|λ|`` is drawn uniformly on the ring
    ``[r_min, r_max]`` and the phase on ``[0, max_phase]``.
    """

    def __init__(
        self,
        size: int,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 2 * math.pi,
        dtype: torch.dtype = torch.float32,
    ):
        """Initializes the transition.

        Args:
            size: the number of modes.
            r_min: the smallest initial magnitude.
            r_max: the largest initial magnitude.
            max_phase: the largest initial phase, in radians.
            dtype: the real data type of ``nu`` and ``θ``.

        Raises:
            ValueError: unless ``0 <= r_min <= r_max <= 1``.
        """
        super().__init__()
        if not 0.0 <= r_min <= r_max <= 1.0:
            raise ValueError(f"Need 0 <= r_min <= r_max <= 1, got {r_min}, {r_max}.")
        magnitude_sq = torch.rand(size, dtype=dtype) * (r_max**2 - r_min**2) + r_min**2
        self.nu = nn.Parameter(torch.log(-0.5 * torch.log(magnitude_sq)))
        self.theta = nn.Parameter(max_phase * torch.rand(size, dtype=dtype))

    def forward(self) -> tuple[Tensor, Tensor]:
        """The multiplier and the input scale.

        Returns:
            ``(λ, sqrt(1 - |λ|²))``, each of shape ``(size,)``.
        """
        lam = torch.polar(torch.exp(-torch.exp(self.nu)), self.theta)
        return lam, torch.sqrt((1 - lam.abs() ** 2).clamp_min(0))


class ZOHDiagonal(nn.Module):
    """S4D's diagonal ``A = -e^a + ib``, discretised by zero-order hold with step ``Δ``.

    From Gu et al., "On the Parameterization and Initialization of Diagonal
    State Space Models" (2022), with the S4D-Lin initialisation
    ``A_n = -1/2 + iπn``. The step ``Δ = e^d`` is learnt per mode, drawn
    log-uniformly in ``[dt_min, dt_max]``.
    """

    def __init__(
        self,
        size: int,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dtype: torch.dtype = torch.float32,
    ):
        """Initializes the transition.

        Args:
            size: the number of modes.
            dt_min: the smallest initial step.
            dt_max: the largest initial step.
            dtype: the real data type of the parameters.

        Raises:
            ValueError: unless ``0 < dt_min <= dt_max``.
        """
        super().__init__()
        if not 0.0 < dt_min <= dt_max:
            raise ValueError(f"Need 0 < dt_min <= dt_max, got {dt_min}, {dt_max}.")
        self.log_real = nn.Parameter(torch.full((size,), math.log(0.5), dtype=dtype))
        self.imag = nn.Parameter(math.pi * torch.arange(size, dtype=dtype))
        low, high = math.log(dt_min), math.log(dt_max)
        self.log_dt = nn.Parameter(low + torch.rand(size, dtype=dtype) * (high - low))

    def forward(self) -> tuple[Tensor, Tensor]:
        """The multiplier and the input scale.

        Returns:
            ``(Ā, (Ā - 1) / A)`` with ``Ā = exp(ΔA)``, each of shape ``(size,)``.
        """
        a = torch.complex(-torch.exp(self.log_real), self.imag)
        dt = torch.exp(self.log_dt)
        # (Ā - 1) / A in complex128: in complex64, small ΔA loses it to cancellation
        a128 = a.to(torch.complex128)
        scale = (torch.exp(dt.to(torch.float64) * a128) - 1) / a128
        return torch.exp(dt * a), scale.to(a.dtype)


class LinearDiagonalCell(Cell):
    """A linear recurrence on a complex diagonal: ``h' = a ⊙ h + s ⊙ (B u)``.

    ``transition()`` gives ``(a, s)``, e.g. :class:`RingDiagonal` (LRU) or
    :class:`ZOHDiagonal` (S4D), and ``input`` is a complex ``B``, e.g. a complex
    :class:`~torchdyno.matrices.Dense`. The state is stored as the real
    ``[Re h; Im h]``, so ``state_size`` is twice the number of modes. Because the
    recurrence is linear, :class:`~torchdyno.layers.Scan` can run it in parallel
    over time.
    """

    def __init__(self, transition: nn.Module, input: LinearMap):
        """Initializes the cell.

        Args:
            transition: a module whose ``forward()`` returns ``(a, s)``.
            input: the complex input matrix ``B``.
        """
        super().__init__()
        self.transition = transition
        self.input = input
        self.modes = input.out_features
        self.state_size = 2 * self.modes

    def coefficients(self, u: Tensor) -> tuple[Tensor, Tensor]:
        """The multiplier and the addend of the recurrence for input ``u``.

        Args:
            u: the input, with ``F`` as the last dimension.

        Returns:
            ``(a, s ⊙ B u)``: ``a`` of shape ``(modes,)``, the addend of shape
            ``u.shape[:-1] + (modes,)``.
        """
        a, s = self.transition()
        return a, s * self.input(u)

    def forward(self, u: Tensor, state: Tensor) -> Tensor:
        """Advance the state by one timestep.

        Args:
            u: the input, shape ``(B, F)``.
            state: ``[Re h; Im h]``, shape ``(B, state_size)``.

        Returns:
            The new state, in the same layout.
        """
        a, b = self.coefficients(u)
        h = a * torch.complex(state[..., : self.modes], state[..., self.modes :]) + b
        return torch.cat([h.real, h.imag], dim=-1)
