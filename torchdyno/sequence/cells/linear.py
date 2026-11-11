"""Linear diagonal recurrences and their transitions (LRU, S4D)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

from torchdyno.matrices.ops import LinearMap
from torchdyno.sequence.cells.base import (
    Cell,
    _width,
)

if TYPE_CHECKING:
    from torch import Tensor

    from torchdyno.matrices import LinearMap


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
        dtype: torch.dtype | None = None,
    ):
        """Initializes the transition.

        Args:
            size: the number of modes.
            r_min: the smallest initial magnitude.
            r_max: the largest initial magnitude.
            max_phase: the largest initial phase, in radians.
            dtype: the real data type of ``nu`` and ``θ``; ``None`` takes
                torch's default.

        Raises:
            ValueError: unless ``0 <= r_min <= r_max <= 1`` and ``r_min < 1``.
        """
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        if not (0.0 <= r_min <= r_max <= 1.0 and r_min < 1.0):
            raise ValueError(
                f"Need 0 <= r_min <= r_max <= 1 with r_min < 1, got {r_min}, {r_max}."
            )
        # in float64, so a draw just below r_max = 1 keeps a finite nu
        draw = torch.rand(size, dtype=dtype).to(torch.float64)
        magnitude_sq = draw * (r_max**2 - r_min**2) + r_min**2
        self.nu = nn.Parameter(torch.log(-0.5 * torch.log(magnitude_sq)).to(dtype))
        self.theta = nn.Parameter(max_phase * torch.rand(size, dtype=dtype))

    def forward(self, u: Tensor | None = None) -> tuple[Tensor, Tensor]:  # noqa: ARG002
        """The multiplier and the input scale.

        Args:
            u: the input; unused, as the transition does not depend on it.

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
        dtype: torch.dtype | None = None,
    ):
        """Initializes the transition.

        Args:
            size: the number of modes.
            dt_min: the smallest initial step.
            dt_max: the largest initial step.
            dtype: the real data type of the parameters; ``None`` takes
                torch's default.

        Raises:
            ValueError: unless ``0 < dt_min <= dt_max``.
        """
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        if not 0.0 < dt_min <= dt_max:
            raise ValueError(f"Need 0 < dt_min <= dt_max, got {dt_min}, {dt_max}.")
        self.log_real = nn.Parameter(torch.full((size,), math.log(0.5), dtype=dtype))
        self.imag = nn.Parameter(math.pi * torch.arange(size, dtype=dtype))
        low, high = math.log(dt_min), math.log(dt_max)
        self.log_dt = nn.Parameter(low + torch.rand(size, dtype=dtype) * (high - low))

    def forward(self, u: Tensor | None = None) -> tuple[Tensor, Tensor]:  # noqa: ARG002
        """The multiplier and the input scale.

        Args:
            u: the input; unused, as the transition does not depend on it.

        Returns:
            ``(Ā, (Ā - 1) / A)`` with ``Ā = exp(ΔA)``, each of shape ``(size,)``.
        """
        a = torch.complex(-torch.exp(self.log_real), self.imag)
        dt = torch.exp(self.log_dt)
        # (Ā - 1) / A by expm1, in complex128: small ΔA loses exp - 1 to cancellation
        a128 = a.to(torch.complex128)
        scale = torch.expm1(dt.to(torch.float64) * a128) / a128
        return torch.exp(dt * a), scale.to(a.dtype)


class LinearDiagonalCell(Cell):
    """A linear recurrence on a diagonal: ``h' = a ⊙ h + s ⊙ (B u)``.

    ``transition(u)`` gives ``(a, s)``, broadcastable to ``u.shape[:-1] +
    (modes,)``: constant, as :class:`RingDiagonal` (LRU) and
    :class:`ZOHDiagonal` (S4D) are, or computed from the input, as a selective
    step is. ``input`` is ``B``. If ``a``, ``s`` or ``B u`` is complex, the
    state is stored as the real ``[Re h; Im h]`` and ``state_size`` is twice
    the number of modes; otherwise it is ``h`` itself. Because the recurrence
    is linear, :class:`~torchdyno.sequence.Scan` can run it in parallel over
    time.
    """

    def __init__(self, transition: nn.Module, input: LinearMap):
        """Initializes the cell.

        Args:
            transition: a module whose ``forward(u)`` returns ``(a, s)``.
            input: the input matrix ``B``.

        Raises:
            ValueError: if the transition has another number of modes than the
                input matrix.
        """
        super().__init__()
        self.transition = transition
        self.input = input
        self.modes = input.out_features
        weight = next(input.parameters(), None)
        dtype = torch.get_default_dtype() if weight is None else weight.dtype
        with torch.no_grad():
            probe = torch.zeros(1, input.in_features, dtype=dtype)
            a, s = transition(probe)
            _width("transition", a.shape[-1], self.modes)
            parts = (a, s, input(probe))
        self.is_complex = any(part.is_complex() for part in parts)
        self.state_size = 2 * self.modes if self.is_complex else self.modes

    def coefficients(self, u: Tensor) -> tuple[Tensor, Tensor]:
        """The multiplier and the addend of the recurrence for input ``u``.

        Args:
            u: the input, with ``F`` as the last dimension.

        Returns:
            ``(a, s ⊙ B u)``: ``a`` broadcastable to the addend, which has shape
            ``u.shape[:-1] + (modes,)``.
        """
        a, s = self.transition(u)
        return a, s * self.input(u)

    def unpack(self, state: Tensor) -> Tensor:
        """The state ``h`` from its stored form."""
        if not self.is_complex:
            return state
        return torch.complex(state[..., : self.modes], state[..., self.modes :])

    def pack(self, h: Tensor) -> Tensor:
        """The stored form of the state ``h``."""
        return torch.cat([h.real, h.imag], dim=-1) if self.is_complex else h

    def forward(self, u: Tensor, state: Tensor) -> Tensor:
        """Advance the state by one timestep.

        Args:
            u: the input, shape ``(B, F)``.
            state: the stored state, shape ``(B, state_size)``.

        Returns:
            The new state, in the same layout.
        """
        a, b = self.coefficients(u)
        return self.pack(a * self.unpack(state) + b)
