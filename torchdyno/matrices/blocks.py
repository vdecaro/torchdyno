"""Block structures: block-diagonal matrices and skew couplings between blocks."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
)

import torch
import torch.nn.functional as F
from torch import nn

from torchdyno.matrices._storage import (
    _draw,
    _sizes,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

    from torch import Tensor

    from torchdyno.matrices.init import Init


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
        dtype: torch.dtype | None = None,
    ):
        """Initializes the matrix.

        Args:
            sizes: the size of each square block.
            init: initialiser of each block; ``None`` uses
                ``torch.nn.init.orthogonal_``.
            bound: applied to the masked entries on every use, e.g.
                ``torch.tanh``; it must map zero to zero.
            dtype: the data type; ``None`` takes torch's default.

        Raises:
            ValueError: if a block size is not positive, or ``bound`` does not
                map zero to zero.
        """
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        _sizes(*sizes)
        if bound is not None and bound(torch.zeros(1, dtype=dtype)).ne(0).any():
            raise ValueError("bound must map zero to zero, or masked entries leak.")
        size = sum(sizes)
        self.in_features = self.out_features = size
        weight = torch.zeros(size, size, dtype=dtype)
        start = 0
        for block in sizes:
            entries = _draw(init or nn.init.orthogonal_, (block, block), dtype)
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
        dtype: torch.dtype | None = None,
    ):
        """Initializes the coupling.

        Args:
            sizes: the size of each block.
            pairs: which blocks to couple: a number of random pairs, a float
                fraction of all pairs, ``"ring"`` to couple each block with the
                next, or an explicit list of ``(i, j)`` pairs.
            init: initialiser of each coupling block; ``None`` uses
                ``torch.nn.init.orthogonal_``.
            dtype: the data type; ``None`` takes torch's default.

        Raises:
            ValueError: if a block size is not positive, or ``pairs`` names
                pairs that do not exist, repeat, or couple a block with itself.
        """
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        _sizes(*sizes)
        size = sum(sizes)
        self.in_features = self.out_features = size
        offsets = [0]
        for width in sizes:
            offsets.append(offsets[-1] + width)
        weight = torch.zeros(size, size, dtype=dtype)
        for i, j in _coupling_pairs(len(sizes), pairs):
            entries = _draw(init or nn.init.orthogonal_, (sizes[i], sizes[j]), dtype)
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
    candidates = [(i, j) for i in range(n_blocks - 1) for j in range(i + 1, n_blocks)]
    if isinstance(pairs, str):
        if pairs != "ring":
            raise ValueError(
                f"pairs must be a number, a list or 'ring', got {pairs!r}."
            )
        return [(i, (i + 1) % n_blocks) for i in range(n_blocks)]
    if isinstance(pairs, list):
        _check_pairs(n_blocks, pairs)
        return pairs
    if isinstance(pairs, float):
        if not 0.0 <= pairs <= 1.0:
            raise ValueError(f"A fraction of pairs must be in [0, 1], got {pairs}.")
        pairs = int(pairs * len(candidates))
    if not 0 <= pairs <= len(candidates):
        raise ValueError(
            f"{n_blocks} blocks have {len(candidates)} pairs, cannot couple {pairs}."
        )
    picked = torch.randperm(len(candidates))[:pairs]
    return [candidates[i] for i in picked.tolist()]


def _check_pairs(n_blocks: int, pairs: list[tuple[int, int]]) -> None:
    """Raise unless ``pairs`` are distinct pairs of distinct, existing blocks."""
    unordered = [tuple(sorted(pair)) for pair in pairs]
    for (i, j), pair in zip(unordered, pairs, strict=True):
        if i == j:
            raise ValueError(f"A block cannot be coupled with itself: {pair}.")
        if not 0 <= i < j < n_blocks:
            raise ValueError(f"No such pair among {n_blocks} blocks: {pair}.")
    if len(set(unordered)) < len(unordered):
        raise ValueError(f"A pair is listed twice: {pairs}.")
