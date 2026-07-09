"""Swappable initialization objects for recurrent cores."""

import math
from typing import Optional, Tuple

import torch
from torch import Generator, Tensor


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

    def __init__(self, r_min: float = 0.0, r_max: float = 1.0,
                 max_phase: float = 2 * math.pi):
        if not (0.0 <= r_min <= r_max):
            raise ValueError(f"require 0 <= r_min <= r_max, got {r_min}, {r_max}.")
        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase

    def __call__(self, n_modes: int, *, generator: Optional[Generator] = None,
                 dtype: torch.dtype = torch.float32) -> Tuple[Tensor, Tensor]:
        u1 = torch.rand(n_modes, generator=generator, dtype=dtype)
        u2 = torch.rand(n_modes, generator=generator, dtype=dtype)
        mag_sq = u1 * (self.r_max**2 - self.r_min**2) + self.r_min**2
        nu = torch.log(-0.5 * torch.log(mag_sq))     # ν = log(−log|λ|)
        theta = self.max_phase * u2
        return nu, theta
