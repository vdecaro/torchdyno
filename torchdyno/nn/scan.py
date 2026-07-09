"""Parallel prefix (associative) scan for first-order linear recurrences."""

import torch
from torch import Tensor


def associative_scan(a: Tensor, b: Tensor) -> Tensor:
    """Inclusive scan of ``h_t = a_t ⊙ h_{t-1} + b_t`` over ``dim=0`` (``h_{-1}=0``).

    Uses a Hillis-Steele scheme: ``⌈log2 T⌉`` vectorized combine steps over the
    whole tensor. The binary operator on ``(a, b)`` segment pairs is
    ``(a_L, b_L) ∘ (a_R, b_R) = (a_R·a_L, a_R·b_L + b_R)``. Works for real or
    complex tensors and is differentiable (no in-place ops).

    Args:
        a: per-step multipliers, shape ``(T, ...)``.
        b: per-step addends, same shape as ``a``.

    Returns:
        The scanned sequence ``h``, same shape as ``b``.
    """
    steps = b.shape[0]
    shift = 1
    while shift < steps:
        # Bind old-tensor slices before rebinding a/b; both combines use them.
        a_cur, a_prev = a[shift:], a[:-shift]  # right (current), left (earlier)
        b_cur, b_prev = b[shift:], b[:-shift]
        a = torch.cat([a[:shift], a_cur * a_prev], dim=0)
        b = torch.cat([b[:shift], a_cur * b_prev + b_cur], dim=0)
        shift *= 2
    return b
