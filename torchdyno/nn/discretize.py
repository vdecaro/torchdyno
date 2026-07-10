"""Discretization of continuous diagonal state-space models."""

import torch
from torch import Tensor


def zoh(A: Tensor, dt: Tensor, B: Tensor) -> tuple[Tensor, Tensor]:
    """Zero-order-hold discretization of a diagonal continuous SSM.

    ``Ā = exp(dt·A)``; ``B̄ = ((Ā − 1)/A)·B`` (elementwise over the diagonal ``A``).
    Requires ``A ≠ 0`` (satisfied by the S4D-Lin/Inv inits, where ``Re(A) = −½``).

    Args:
        A: continuous diagonal state matrix, ``(N,)`` complex (``Re(A) < 0``).
        dt: per-mode timestep, ``(N,)`` real ``> 0``.
        B: complex input matrix, ``(N, input_size)``.

    Returns:
        ``(A_bar: (N,) complex, B_bar: (N, input_size) complex)``.
    """
    a_bar = torch.exp(dt * A)
    # expm1(dt·A) == a_bar - 1 mathematically, but avoids the catastrophic
    # cancellation that a naive subtraction incurs for small dt·A.
    b_bar = (torch.expm1(dt * A) / A).unsqueeze(-1) * B
    return a_bar, b_bar
