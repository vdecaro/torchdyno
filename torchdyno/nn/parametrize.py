"""Reparameterizations mapping unconstrained surrogates to effective params."""

import torch
from torch import Tensor, nn


class StableExpComplex(nn.Module):
    """Map real surrogates ``(ν, θ)`` to a stable complex diagonal eigenvalue.

    ``λ = exp(−exp(ν) + iθ)`` so ``|λ| = exp(−exp(ν)) ∈ (0, 1)`` for every real
    ``ν``. Stability is thus a property of the parameterization: ``.parameters()``
    exposes only ``ν, θ`` and any optimizer is automatically safe against
    ``|λ| ≥ 1``. Use with two originals via
    ``torch.nn.utils.parametrize.register_parametrization``.
    """

    def forward(self, nu: Tensor, theta: Tensor) -> Tensor:
        return torch.polar(torch.exp(-torch.exp(nu)), theta)

    def right_inverse(self, lam: Tensor) -> tuple[Tensor, Tensor]:
        tiny = torch.finfo(lam.real.dtype).tiny
        mag = lam.abs().clamp_min(tiny)
        nu = torch.log(-torch.log(mag))
        theta = torch.angle(lam)
        return nu, theta
