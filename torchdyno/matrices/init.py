"""Initialisers: callables that fill a tensor in place, as ``torch.nn.init``."""

from __future__ import annotations

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


def uniform(scale: float) -> Init:
    """Initialise from ``U(-scale, scale)``; complex parts are drawn independently.

    Args:
        scale: the half-width of the interval.

    Returns:
        An in-place initialiser.
    """

    def init(tensor: Tensor) -> Tensor:
        nn.init.uniform_(_real(tensor), -scale, scale)
        return tensor

    return init


def normal(std: float) -> Init:
    """Initialise from ``N(0, std²)``; complex parts are drawn independently.

    Args:
        std: the standard deviation of each part.

    Returns:
        An in-place initialiser.
    """

    def init(tensor: Tensor) -> Tensor:
        nn.init.normal_(_real(tensor), 0.0, std)
        return tensor

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
