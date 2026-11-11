"""Matrices: how a weight matrix is structured, initialised and applied.

Every matrix is a module whose ``forward(x)`` applies ``x Wᵀ`` without building
``W`` when its structure allows it, and whose ``dense()`` returns ``W``. A
matrix's ``weight`` is its storage, which can differ from ``W``: masked,
bounded, or, for a complex matrix, the real and imaginary parts in a last
dimension of 2, so that casts such as ``.double()`` reach both parts.
``torch.nn.Linear`` works wherever a matrix is expected. Initialisers follow the
``torch.nn.init`` convention, callables that fill a tensor in place, so every
``torch.nn.init`` function can be passed directly. Freeze a matrix with
``requires_grad_(False)``; freezing never changes its initialisation.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    Protocol,
)

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterator,
    )

    from torch import Tensor

    Init = Callable[[Tensor], Tensor]


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


def _real(tensor: Tensor) -> Tensor:
    """A real view of ``tensor``: the real and imaginary parts if it is complex."""
    return torch.view_as_real(tensor) if tensor.is_complex() else tensor


def _parameter(value: Tensor) -> nn.Parameter:
    """A parameter storing ``value``; a complex one as its real and imaginary parts."""
    return nn.Parameter(_real(value).clone() if value.is_complex() else value)


def _value(stored: Tensor, is_complex: bool) -> Tensor:
    """The value a :func:`_parameter` stores."""
    return torch.view_as_complex(stored) if is_complex else stored


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
        dtype: torch.dtype = torch.float32,
    ):
        """Initializes the matrix.

        Args:
            in_features: the input width.
            out_features: the output width.
            init: initialiser of ``W``; ``None`` draws from
                ``U(±1/sqrt(in_features))`` like ``torch.nn.Linear``.
            bias: initialiser of the bias, or ``None`` for no bias.
            dtype: the data type.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_complex = dtype.is_complex
        weight = torch.empty(out_features, in_features, dtype=dtype)
        (init or uniform(in_features**-0.5))(weight)
        self.weight = _parameter(weight)
        self.bias = None
        if bias is not None:
            offset = torch.empty(out_features, dtype=dtype)
            bias(offset)
            self.bias = _parameter(offset)

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


class Diagonal(nn.Module):
    """A diagonal matrix, optionally bounded elementwise: ``x ⊙ bound(d)``."""

    def __init__(
        self,
        size: int,
        init: Init | None = None,
        bound: Callable[[Tensor], Tensor] | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Initializes the matrix.

        Args:
            size: the number of diagonal entries.
            init: initialiser of the stored entries; ``None`` draws from
                ``U(-1, 1)``.
            bound: applied to the stored entries on every use, e.g.
                ``torch.tanh`` to keep them in ``(-1, 1)``.
            dtype: the data type.
        """
        super().__init__()
        self.in_features = self.out_features = size
        self.is_complex = dtype.is_complex
        weight = torch.empty(size, dtype=dtype)
        (init or uniform(1.0))(weight)
        self.weight = _parameter(weight)
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


class BlockDiagonal(nn.Module):
    """A block-diagonal matrix whose zero entries stay zero.

    The mask is the nonzero pattern of the initial blocks, so blocks drawn with
    a sparse initialiser such as ``torch.nn.init.sparse_`` stay sparse during
    training.
    """

    mask: Tensor

    def __init__(
        self,
        sizes: list[int],
        init: Init | None = None,
        bound: Callable[[Tensor], Tensor] | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Initializes the matrix.

        Args:
            sizes: the size of each square block.
            init: initialiser of each block; ``None`` uses
                ``torch.nn.init.orthogonal_``.
            bound: applied to the masked entries on every use, e.g.
                ``torch.tanh``; it must map zero to zero.
            dtype: the data type.
        """
        super().__init__()
        size = sum(sizes)
        self.in_features = self.out_features = size
        weight = torch.zeros(size, size, dtype=dtype)
        start = 0
        for block in sizes:
            entries = torch.empty(block, block, dtype=dtype)
            (init or nn.init.orthogonal_)(entries)
            weight[start : start + block, start : start + block] = entries
            start += block
        self.weight = nn.Parameter(weight)
        self.register_buffer("mask", weight != 0)
        self.bound = bound

    def dense(self) -> Tensor:
        """The full matrix, masked and bounded."""
        weight = self.mask * self.weight
        return weight if self.bound is None else self.bound(weight)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the matrix.

        Args:
            x: input with the total block size as the last dimension.

        Returns:
            ``x Wᵀ``.
        """
        # shortcut: multiplies by the masked dense matrix; switch to per-block
        # matmuls if the state grows past a few thousand units
        return F.linear(x, self.dense())


class SkewCoupling(nn.Module):
    """A skew-symmetric coupling between the blocks of a block-diagonal state.

    Only the coupled pairs are stored and trained; ``dense()`` is ``C - Cᵀ``
    over them, so the coupling stays skew-symmetric during training.
    """

    mask: Tensor

    def __init__(
        self,
        sizes: list[int],
        pairs: int | float | Literal["ring"] | list[tuple[int, int]],
        init: Init | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Initializes the coupling.

        Args:
            sizes: the size of each block.
            pairs: which blocks to couple: a number of random pairs, a float
                fraction of all pairs, ``"ring"`` to couple each block with the
                next, or an explicit list of ``(i, j)`` pairs.
            init: initialiser of each coupling block; ``None`` uses
                ``torch.nn.init.orthogonal_``.
            dtype: the data type.

        Raises:
            ValueError: if a pair couples a block with itself.
        """
        super().__init__()
        size = sum(sizes)
        self.in_features = self.out_features = size
        offsets = [0]
        for width in sizes:
            offsets.append(offsets[-1] + width)
        weight = torch.zeros(size, size, dtype=dtype)
        for i, j in _coupling_pairs(len(sizes), pairs):
            if i == j:
                raise ValueError(f"A block cannot be coupled with itself: {i}.")
            entries = torch.empty(sizes[i], sizes[j], dtype=dtype)
            (init or nn.init.orthogonal_)(entries)
            row, col = (i, j) if i < j else (j, i)
            block = entries if i < j else entries.T
            weight[offsets[row] : offsets[row + 1], offsets[col] : offsets[col + 1]] = (
                block
            )
        self.weight = nn.Parameter(weight)
        self.register_buffer("mask", weight != 0)

    def dense(self) -> Tensor:
        """The skew-symmetric matrix ``C - Cᵀ``."""
        coupled = self.mask * self.weight
        return coupled - coupled.T

    def forward(self, x: Tensor) -> Tensor:
        """Apply the coupling.

        Args:
            x: input with the total block size as the last dimension.

        Returns:
            ``x Lᵀ`` with ``L = C - Cᵀ``.
        """
        return F.linear(x, self.dense())


def _coupling_pairs(
    n_blocks: int, pairs: int | float | Literal["ring"] | list[tuple[int, int]]
) -> list[tuple[int, int]]:
    if pairs == "ring":
        return [(i, (i + 1) % n_blocks) for i in range(n_blocks)]
    if isinstance(pairs, list):
        return pairs
    candidates = [(i, j) for i in range(n_blocks - 1) for j in range(i + 1, n_blocks)]
    count = int(pairs * len(candidates)) if isinstance(pairs, float) else int(pairs)
    picked = torch.randperm(len(candidates))[: min(count, len(candidates))]
    return [candidates[i] for i in picked.tolist()]
