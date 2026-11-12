"""Initialisers: callables that fill a tensor in place, as ``torch.nn.init``."""

from __future__ import annotations

import math
from typing import (
    TYPE_CHECKING,
)

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

    from torch import Tensor

    Init = Callable[[Tensor], Tensor]


def _real(tensor: Tensor) -> Tensor:
    """A real view of ``tensor``: the real and imaginary parts if it is complex."""
    return torch.view_as_real(tensor) if tensor.is_complex() else tensor


def uniform(scale: float | Tensor) -> Init:
    """Initialise from ``U(-scale, scale)``; complex parts are drawn independently.

    Args:
        scale: the half-width of the interval, or a tensor of them that
            broadcasts to the matrix, e.g. one per input column to scale each
            input channel.

    Returns:
        An in-place initialiser.
    """

    def init(tensor: Tensor) -> Tensor:
        if isinstance(scale, torch.Tensor):
            nn.init.uniform_(_real(tensor), -1.0, 1.0)
            return tensor.mul_(scale.to(tensor.device))
        nn.init.uniform_(_real(tensor), -scale, scale)
        return tensor

    return init


def normal(std: float | Tensor) -> Init:
    """Initialise from ``N(0, std²)``; complex parts are drawn independently.

    Args:
        std: the standard deviation of each part, or a tensor of them that
            broadcasts to the matrix, e.g. one per input column.

    Returns:
        An in-place initialiser.
    """

    def init(tensor: Tensor) -> Tensor:
        if isinstance(std, torch.Tensor):
            nn.init.normal_(_real(tensor))
            return tensor.mul_(std.to(tensor.device))
        nn.init.normal_(_real(tensor), 0.0, std)
        return tensor

    return init


def polar(
    magnitude: tuple[float, float], phase: tuple[float, float] = (0.0, 2 * math.pi)
) -> Init:
    """Initialise complex entries as ``r e^(iθ)``, ``r`` and ``θ`` uniform in ranges.

    E.g. the eigenvalues of a complex diagonal reservoir, as in Pinna et al.,
    "ParalESN: Enabling parallel information processing in Reservoir Computing"
    (2026): ``magnitude[1] < 1`` keeps every mode stable. The modulus is uniform,
    not the area, so draws are denser near the centre of the disc than on a
    uniform disc.

    Args:
        magnitude: the range ``(low, high)`` of the modulus ``r``.
        phase: the range ``(low, high)`` of the phase ``θ``, in radians.

    Returns:
        An in-place initialiser, which raises ``TypeError`` on a real tensor.

    Raises:
        ValueError: unless ``0 <= magnitude[0] <= magnitude[1]`` and
            ``phase[0] <= phase[1]``.
    """
    if not 0.0 <= magnitude[0] <= magnitude[1]:
        raise ValueError(f"Need 0 <= low <= high for magnitude, got {magnitude}.")
    if not phase[0] <= phase[1]:
        raise ValueError(f"Need low <= high for phase, got {phase}.")

    def init(tensor: Tensor) -> Tensor:
        if not tensor.is_complex():
            raise TypeError(f"polar() draws complex entries, got {tensor.dtype}.")
        with torch.no_grad():
            r = torch.empty_like(tensor.real).uniform_(*magnitude)
            theta = torch.empty_like(r).uniform_(*phase)
            return tensor.copy_(torch.polar(r, theta))

    return init


def cycle(weight: float) -> Init:
    """Initialise a square matrix as a ring: unit ``i`` feeds unit ``i + 1 mod n``.

    The simple cycle reservoir of Rodan and Tino, "Minimum complexity echo state
    network" (2011): every connection has the same ``weight``, which is also
    the spectral radius.

    Args:
        weight: the weight of every connection in the ring.

    Returns:
        An in-place initialiser.
    """

    def init(tensor: Tensor) -> Tensor:
        with torch.no_grad():
            tensor.zero_()
            ring = torch.arange(tensor.shape[0], device=tensor.device)
            return tensor.index_put_((ring.roll(-1), ring), tensor.new_tensor(weight))

    return init


def spectral(init: Init, rho: float) -> Init:
    """Apply ``init``, then rescale the square matrix to spectral radius ``rho``.

    Args:
        init: the initialiser to rescale.
        rho: the target spectral radius.

    Returns:
        An in-place initialiser, which raises ``ValueError`` if ``init`` draws a
        matrix whose eigenvalues are all zero, such as a delay line.

    Raises:
        ValueError: if ``rho`` is not positive.
    """
    if not rho > 0:
        raise ValueError(f"rho must be > 0, got {rho}.")

    def rescaled(tensor: Tensor) -> Tensor:
        init(tensor)
        with torch.no_grad():
            radius = torch.linalg.eigvals(tensor).abs().max()
            if radius == 0:
                raise ValueError(
                    "Every eigenvalue of the initial matrix is zero, so no scale "
                    "gives it a spectral radius."
                )
            return tensor.mul_(rho / radius)

    return rescaled
