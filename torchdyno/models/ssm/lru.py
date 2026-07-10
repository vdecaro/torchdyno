"""LRUCore: a Linear Recurrent Unit as a SequenceCore.

Orvieto et al. (2023), *Resurrecting Recurrent Neural Networks for Long Sequences* (ICML).
A complex diagonal linear recurrence with a stable exponential parameterization and ring
initialization, run by a parallel associative scan (via :class:`DiagonalSSMCore`).
"""

from typing import (
    Optional,
    Tuple,
    Union,
)

import torch
import torch.nn.utils.parametrize as P
from torch import (
    Tensor,
    nn,
)

from torchdyno.models.ssm.base import DiagonalSSMCore
from torchdyno.nn.init import Ring
from torchdyno.nn.parametrize import StableExpComplex
from torchdyno.registry import (
    ModelCard,
    register_core,
)

_INITS = {"ring": Ring}


def _resolve_init(init: Union[str, Ring]) -> Ring:
    if isinstance(init, str):
        if init not in _INITS:
            raise ValueError(f"Unknown init {init!r}. Available: {sorted(_INITS)}.")
        return _INITS[init]()
    return init


_LRU_CARD = ModelCard(
    name="lru",
    family="ssm",
    paper="Orvieto, A., Smith, S. L., Gu, A., Fernando, A., Gulcehre, C., "
    "Pascanu, R., & De, S. (2023). Resurrecting Recurrent Neural Networks for "
    "Long Sequences. ICML.",
    description="A Linear Recurrent Unit: a complex diagonal linear state-space "
    "recurrence with a stable exponential parameterization and ring "
    "initialization, run by a parallel associative scan. Frozen (ridge) or "
    "trained (backprop).",
    admits=("ridge", "backprop"),
    adapters=(),
    tasks=("forecast", "classify"),
    default_config={"input_size": 1, "n_modes": 64},
)


@register_core("lru", card=_LRU_CARD)
class LRUCore(DiagonalSSMCore):
    """A Linear Recurrent Unit as a :class:`SequenceCore`."""

    def __init__(
        self,
        input_size: int,
        n_modes: int = 64,
        *,
        trainable: bool = True,
        normalize: bool = True,
        init: Union[str, Ring] = "ring",
        generator: Optional[torch.Generator] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(input_size, n_modes, trainable=trainable, dtype=dtype)
        # RNG order preserved: Ring draws (ν, θ) first, then _make_B draws B — exactly as
        # before the base extraction, so a fixed seed yields byte-identical init.
        nu0, theta0 = _resolve_init(init)(n_modes, generator=generator, dtype=dtype)
        self.lambda_ = nn.Parameter(torch.polar(torch.exp(-torch.exp(nu0)), theta0))
        P.register_parametrization(self, "lambda_", StableExpComplex())
        with torch.no_grad():
            self.parametrizations.lambda_.original0.copy_(nu0)
            self.parametrizations.lambda_.original1.copy_(theta0)
        self.normalize = normalize
        self._make_B(generator, dtype)
        self._finalize()

    def _recurrence(self) -> Tuple[Tensor, Tensor]:
        lam = self.lambda_
        B = torch.complex(self.B_re, self.B_im)
        if self.normalize:
            gamma = torch.sqrt((1.0 - lam.abs() ** 2).clamp_min(0.0))
            B = gamma.unsqueeze(-1) * B
        return lam, B
