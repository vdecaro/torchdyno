"""What every matrix provides, and reading and updating any matrix."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Protocol,
    cast,
)

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
    )

    from torch import Tensor


class LinearMap(Protocol):
    """A module that applies a linear map ``x Wᵀ``: what cells take as a matrix.

    Every matrix in this module satisfies it, and so does ``torch.nn.Linear``.
    """

    in_features: int
    out_features: int

    def __call__(self, x: Tensor) -> Tensor:
        """Apply the map to the last dimension of ``x``."""
        ...

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """The parameters the map is built from."""
        ...


class _Dense(Protocol):
    def dense(self) -> Tensor: ...


def to_dense(matrix: nn.Module) -> Tensor:
    """The matrix ``W`` of a linear map: its ``dense()``, or an ``nn.Linear``'s weight.

    Args:
        matrix: a matrix of this module, or an ``nn.Linear``.

    Returns:
        The ``(out_features, in_features)`` matrix.
    """
    if isinstance(matrix, nn.Linear):
        return matrix.weight
    return cast("_Dense", matrix).dense()


@torch.no_grad()
def spectral_radius(matrix: nn.Module) -> float:
    """The largest absolute eigenvalue of a square matrix's ``W``.

    Args:
        matrix: a matrix of this module, or an ``nn.Linear``.

    Returns:
        The spectral radius, computed in double precision.
    """
    w = to_dense(matrix)
    w = w.to(torch.complex128 if w.is_complex() else torch.float64)
    return torch.linalg.eigvals(w).abs().max().item()


@torch.no_grad()
def update_(matrix: nn.Module, delta: Tensor) -> None:
    """Add ``delta`` to ``W`` as far as the matrix's structure allows, in place.

    For local learning rules, e.g. Hebbian ones, that compute an update of the
    dense ``W``. The update is pulled back to the matrix's parameters through
    ``dense()`` (``Jᵀ delta``), so a diagonal takes the diagonal of ``delta``, a
    block-diagonal matrix its blocks and a skew coupling its skew part, and the
    structure survives. Frozen parameters are updated too.

    Args:
        matrix: a matrix of this module, or an ``nn.Linear``.
        delta: the ``(out_features, in_features)`` update of ``W``.
    """
    params = list(matrix.parameters())
    flags = [p.requires_grad for p in params]
    try:
        for p in params:
            p.requires_grad_(True)
        with torch.enable_grad():
            grads = torch.autograd.grad(
                to_dense(matrix), params, grad_outputs=delta, allow_unused=True
            )
    finally:
        for p, flag in zip(params, flags, strict=True):
            p.requires_grad_(flag)
    for p, grad in zip(params, grads, strict=True):
        if grad is not None:
            p.add_(grad)
