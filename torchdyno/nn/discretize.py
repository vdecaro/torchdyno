"""Discretization of continuous diagonal state-space models."""

import torch
from torch import Tensor


def zoh(A: Tensor, dt: Tensor, B: Tensor) -> tuple[Tensor, Tensor]:
    """Zero-order-hold discretization of a diagonal continuous SSM.

    ``Ā = exp(dt·A)``; ``B̄ = ((Ā − 1)/A)·B`` (elementwise; the subtraction is done in
    complex128 for numerical stability at small ``dt·A``).
    Requires ``A ≠ 0`` (satisfied by the S4D-Lin/Inv inits, where ``Re(A) = −½``).

    Args:
        A: continuous diagonal state matrix, ``(N,)`` complex (``Re(A) < 0``).
        dt: per-mode timestep, ``(N,)`` real ``> 0``.
        B: complex input matrix, ``(N, input_size)``.

    Returns:
        ``(A_bar: (N,) complex, B_bar: (N, input_size) complex)``.
    """
    a_bar = torch.exp(dt * A)
    # Stable (Ā − 1)/A: do the subtraction in complex128 so small dt·A does not lose
    # precision to float32 catastrophic cancellation, using only complex-supported
    # exp/sub/div/cast — keeps the library torch-version-agnostic (avoids complex
    # torch.expm1, which requires torch >= 2.1).
    a128 = A.to(torch.complex128)
    a_hi = torch.exp(dt.to(torch.float64) * a128)
    b_bar = ((a_hi - 1.0) / a128).to(A.dtype).unsqueeze(-1) * B
    return a_bar, b_bar
