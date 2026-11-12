"""How matrices store their parameters."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

import torch
from torch import nn

from torchdyno.matrices.init import _real

if TYPE_CHECKING:
    from torch import Tensor

    from torchdyno.matrices.init import Init


def _parameter(value: Tensor) -> nn.Parameter:
    """A parameter storing ``value``; a complex one as its real and imaginary parts."""
    return nn.Parameter(_real(value).clone() if value.is_complex() else value)


def _value(stored: Tensor, is_complex: bool) -> Tensor:
    """The value a ``_parameter()`` stores."""
    return torch.view_as_complex(stored) if is_complex else stored


def _draw(init: Init, shape: tuple[int, ...], dtype: torch.dtype) -> Tensor:
    """A tensor filled by ``init``, drawn in at least single precision.

    Torch has no QR or eigenvalue kernels for half precision, which
    ``torch.nn.init.orthogonal_`` and [`spectral()`][torchdyno.matrices.spectral] need.
    """
    tensor = torch.empty(shape, dtype=torch.promote_types(dtype, torch.float32))
    init(tensor)
    return tensor.to(dtype)


def _sizes(*sizes: int) -> None:
    """Raise unless every size is positive."""
    if not sizes or min(sizes) < 1:
        raise ValueError(f"Sizes must be positive, got {list(sizes)}.")
