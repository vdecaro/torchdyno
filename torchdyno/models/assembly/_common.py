"""Shared construction helpers for coupled-modular assembly cores."""

from typing import (
    Callable,
    List,
    Literal,
    Tuple,
    Union,
)

import torch
from torch import Tensor

from torchdyno.models import initializers
from torchdyno.models.rnn_assembly import SkewAntisymmetricCoupling
from torchdyno.models.rnn_assembly.skew_symm_coupling import get_coupling_indices


def build_coupling(
    block_sizes: List[int],
    coupling_topology: Union[int, float, Literal["ring"], List[Tuple[int, int]]],
    coupling_block_init: Union[str, Callable[..., Tensor]] = "orthogonal",
    dtype: torch.dtype = torch.float32,
) -> SkewAntisymmetricCoupling:
    """Build a skew-symmetric inter-module coupling for an assembly.

    Args:
        block_sizes: hidden size of each module.
        coupling_topology: which module pairs are coupled — an int/float count
            or fraction, ``"ring"``, or an explicit list of ``(i, j)`` pairs.
        coupling_block_init: initializer for each coupling block, either a name
            in ``torchdyno.models.initializers`` or a callable ``(shape, dtype)``.
        dtype: dtype of the coupling blocks.

    Returns:
        A configured :class:`SkewAntisymmetricCoupling` whose ``couplings``
        property is a skew-symmetric ``(H, H)`` matrix, ``H = sum(block_sizes)``.
    """
    if isinstance(coupling_block_init, str):
        init_fn: Callable[..., Tensor] = getattr(initializers, coupling_block_init)
    else:
        init_fn = coupling_block_init

    indices = get_coupling_indices(block_sizes, coupling_topology)
    coupling_blocks = [
        init_fn((block_sizes[i], block_sizes[j]), dtype=dtype) for i, j in indices
    ]
    return SkewAntisymmetricCoupling(block_sizes, coupling_blocks, indices)
