"""Dense: a dense matrix with an optional bias."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

import torch
import torch.nn.functional as F
from torch import nn

from torchdyno.matrices._storage import (
    _draw,
    _parameter,
    _sizes,
    _value,
)
from torchdyno.matrices.init import uniform

if TYPE_CHECKING:
    from torch import Tensor

    from torchdyno.matrices.init import Init


class Dense(nn.Module):
    """A dense matrix with an optional bias: ``x Wᵀ + b``.

    With a complex ``dtype``, real inputs are cast to complex before the product.
    """

    bias: nn.Parameter | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init: Init | None = None,
        bias: Init | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initializes the matrix.

        Args:
            in_features: the input width.
            out_features: the output width.
            init: initialiser of ``W``; ``None`` draws from
                ``U(±1/sqrt(in_features))`` like ``torch.nn.Linear``.
            bias: initialiser of the bias, or ``None`` for no bias.
            dtype: the data type; ``None`` takes torch's default.

        Raises:
            ValueError: if a width is not positive.
        """
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        _sizes(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.is_complex = dtype.is_complex
        weight = _draw(
            init or uniform(in_features**-0.5), (out_features, in_features), dtype
        )
        self.weight = _parameter(weight)
        self.bias = None
        if bias is not None:
            self.bias = _parameter(_draw(bias, (out_features,), dtype))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the matrix.

        Args:
            x: input with ``in_features`` as the last dimension.

        Returns:
            ``x Wᵀ + b``.
        """
        weight = self.dense()
        bias = None if self.bias is None else _value(self.bias, self.is_complex)
        if self.is_complex and not x.is_complex():
            x = x.to(weight.dtype)
        return F.linear(x, weight, bias)

    def dense(self) -> Tensor:
        """The ``(out_features, in_features)`` matrix ``W``."""
        return _value(self.weight, self.is_complex)
