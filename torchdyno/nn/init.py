"""Swappable initialization objects for recurrent cores."""

import math
from typing import (
    Optional,
    Tuple,
)

import torch
from torch import (
    Generator,
    Tensor,
)


class Ring:
    """Ring initialization of complex diagonal eigenvalues (Orvieto et al. 2023).

    Samples eigenvalue magnitudes uniformly on the annulus ``[r_min, r_max]`` and
    phases uniformly on ``[0, max_phase]``, returned as the ``(ν, θ)`` surrogates
    of :class:`~torchdyno.nn.parametrize.StableExpComplex`.

    Args:
        r_min: inner magnitude bound (``0 ≤ r_min ≤ r_max``).
        r_max: outer magnitude bound (``≤ 1``).
        max_phase: upper phase bound (radians).
    """

    def __init__(
        self, r_min: float = 0.0, r_max: float = 1.0, max_phase: float = 2 * math.pi
    ):
        if not (0.0 <= r_min <= r_max <= 1.0):
            raise ValueError(f"require 0 <= r_min <= r_max <= 1, got {r_min}, {r_max}.")
        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase

    def __call__(
        self,
        n_modes: int,
        *,
        generator: Optional[Generator] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[Tensor, Tensor]:
        u1 = torch.rand(n_modes, generator=generator, dtype=dtype)
        u2 = torch.rand(n_modes, generator=generator, dtype=dtype)
        mag_sq = u1 * (self.r_max**2 - self.r_min**2) + self.r_min**2
        nu = torch.log(-0.5 * torch.log(mag_sq))  # ν = log(−log|λ|)
        theta = self.max_phase * u2
        return nu, theta


class S4DLin:
    """S4D-Lin initialization (Gu et al. 2022): ``A_n = −1/2 + i·π·n``.

    Returns the surrogates ``(log_Aʳᵉ, Aⁱᵐ, log_dt)`` for
    :class:`~torchdyno.models.ssm.s4d.S4DCore`, where ``A = −exp(log_Aʳᵉ) + i·Aⁱᵐ`` and
    ``Δ = exp(log_dt) ∈ [dt_min, dt_max]``.

    Args:
        dt_min: lower timestep bound (``0 < dt_min ≤ dt_max``).
        dt_max: upper timestep bound.
    """

    def __init__(self, dt_min: float = 1e-3, dt_max: float = 1e-1):
        if not (0.0 < dt_min <= dt_max):
            raise ValueError(f"require 0 < dt_min <= dt_max, got {dt_min}, {dt_max}.")
        self.dt_min = dt_min
        self.dt_max = dt_max

    def _a_imag(self, n_modes: int, dtype: torch.dtype) -> Tensor:
        n = torch.arange(n_modes, dtype=dtype)
        return math.pi * n

    def __call__(
        self,
        n_modes: int,
        *,
        generator: Optional[Generator] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        log_are = torch.log(torch.full((n_modes,), 0.5, dtype=dtype))  # Re(A) = −½
        a_im = self._a_imag(n_modes, dtype)
        u = torch.rand(n_modes, generator=generator, dtype=dtype)
        lo, hi = math.log(self.dt_min), math.log(self.dt_max)
        log_dt = lo + u * (hi - lo)
        return log_are, a_im, log_dt


class S4DInv(S4DLin):
    """S4D-Inv initialization: ``A_n = −1/2 + i·(N/π)·(N/(2n+1) − 1)``."""

    def _a_imag(self, n_modes: int, dtype: torch.dtype) -> Tensor:
        n = torch.arange(n_modes, dtype=dtype)
        return (n_modes / math.pi) * (n_modes / (2 * n + 1) - 1)
