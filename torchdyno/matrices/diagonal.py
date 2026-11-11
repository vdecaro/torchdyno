"""Diagonal: a diagonal matrix, optionally bounded."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

import torch
from torch import nn

from torchdyno.matrices._storage import (
    _draw,
    _parameter,
    _sizes,
    _value,
)
from torchdyno.matrices.init import uniform

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

    from torch import Tensor

    from torchdyno.matrices.init import Init


class Diagonal(nn.Module):
    """A diagonal matrix, optionally bounded elementwise: ``x ⊙ bound(d)``."""

    def __init__(
        self,
        size: int,
        init: Init | None = None,
        bound: Callable[[Tensor], Tensor] | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initializes the matrix.

        Args:
            size: the number of diagonal entries.
            init: initialiser of the stored entries; ``None`` draws from
                ``U(-1, 1)``.
            bound: applied to the stored entries on every use, e.g.
                ``torch.tanh`` to keep them in ``(-1, 1)``.
            dtype: the data type; ``None`` takes torch's default.

        Raises:
            ValueError: if ``size`` is not positive.
        """
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        _sizes(size)
        self.in_features = self.out_features = size
        self.is_complex = dtype.is_complex
        self.weight = _parameter(_draw(init or uniform(1.0), (size,), dtype))
        self.bound = bound

    def diagonal(self) -> Tensor:
        """The diagonal entries, after the bound."""
        entries = _value(self.weight, self.is_complex)
        return entries if self.bound is None else self.bound(entries)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the matrix.

        Args:
            x: input with ``size`` as the last dimension.

        Returns:
            ``x`` scaled entrywise by the diagonal.
        """
        return x * self.diagonal()

    def dense(self) -> Tensor:
        """The ``(size, size)`` matrix."""
        return torch.diag(self.diagonal())
