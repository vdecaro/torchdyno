"""Reparameterizations mapping unconstrained surrogates to effective params."""

import torch
from torch import (
    Tensor,
    nn,
)


class StableExpComplex(nn.Module):
    """Map real surrogates ``(ν, θ)`` to a stable complex diagonal eigenvalue.

    ``λ = exp(−exp(ν) + iθ)`` so ``|λ| = exp(−exp(ν)) ∈ (0, 1)`` for every real
    ``ν``. Stability is thus a property of the parameterization: ``.parameters()``
    exposes only ``ν, θ`` and any optimizer is automatically safe against
    ``|λ| ≥ 1``. Use with two originals via
    ``torch.nn.utils.parametrize.register_parametrization``.
    """

    def forward(self, nu: Tensor, theta: Tensor) -> Tensor:
        """Map real surrogates ``(ν, θ)`` to the effective complex eigenvalue ``λ``."""
        return torch.polar(torch.exp(-torch.exp(nu)), theta)

    def right_inverse(self, lam: Tensor) -> tuple[Tensor, Tensor]:
        """Recover the ``(ν, θ)`` surrogates from an effective ``λ`` (θ modulo 2π)."""
        tiny = torch.finfo(lam.real.dtype).tiny
        mag = lam.abs().clamp_min(tiny)
        # Also guard the |λ|→1 tail: when exp(ν) underflows to 0, mag rounds to
        # 1.0 and -log(mag)=0 would make the outer log -inf. clamp_min keeps ν finite.
        nu = torch.log((-torch.log(mag)).clamp_min(tiny))
        theta = torch.angle(lam)
        return nu, theta


class NegExpComplex(nn.Module):
    """Map real surrogates ``(log_re, im)`` to a stable continuous eigenvalue.

    ``A = −exp(log_re) + i·im`` so ``Re(A) = −exp(log_re) < 0`` for every real
    ``log_re``. Continuous-time stability (``Re(A) < 0``) is thus a property of the
    parameterization; ``.parameters()`` exposes only the two real surrogates. Use with
    two originals via ``torch.nn.utils.parametrize.register_parametrization``.
    """

    def forward(self, log_re: Tensor, im: Tensor) -> Tensor:
        """Map ``(log_re, im)`` to the effective continuous eigenvalue ``A``."""
        return torch.complex(-torch.exp(log_re), im)

    def right_inverse(self, a: Tensor) -> tuple[Tensor, Tensor]:
        """Recover ``(log_re, im)`` from an effective ``A`` (exact; guards ``Re(A) → 0⁻``)."""
        tiny = torch.finfo(a.real.dtype).tiny
        return torch.log((-a.real).clamp_min(tiny)), a.imag
