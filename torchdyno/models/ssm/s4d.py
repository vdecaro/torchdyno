"""S4DCore: the diagonal S4D state-space model as a SequenceCore.

Gu, Gupta, Goel & Ré (2022), *On the Parameterization and Initialization of Diagonal State
Space Models* (NeurIPS). A continuous-time complex diagonal system with HiPPO (S4D-Lin/Inv)
initialization and a learnable per-mode timestep, ZOH-discretized and run by a parallel
associative scan (via :class:`DiagonalSSMCore`).
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
from torchdyno.nn.discretize import zoh
from torchdyno.nn.init import (
    S4DInv,
    S4DLin,
)
from torchdyno.nn.parametrize import NegExpComplex
from torchdyno.registry import (
    ModelCard,
    register_core,
)

_INITS = {"s4d-lin": S4DLin, "s4d-inv": S4DInv}


def _resolve_init(init: Union[str, S4DLin]) -> S4DLin:
    if isinstance(init, str):
        if init not in _INITS:
            raise ValueError(f"Unknown init {init!r}. Available: {sorted(_INITS)}.")
        return _INITS[init]()
    return init


_S4D_CARD = ModelCard(
    name="s4d",
    family="ssm",
    paper="Gu, A., Gupta, A., Goel, K., & Ré, C. (2022). On the Parameterization "
    "and Initialization of Diagonal State Space Models. NeurIPS.",
    description="S4D: a diagonal state-space model — a continuous-time complex "
    "diagonal system with HiPPO (S4D-Lin/Inv) initialization and a learnable "
    "per-mode timestep, ZOH-discretized and run by a parallel associative scan. "
    "Frozen (ridge) or trained (backprop).",
    admits=("ridge", "backprop"),
    adapters=(),
    tasks=("forecast", "classify"),
    default_config={"input_size": 1, "n_modes": 64},
)


@register_core("s4d", card=_S4D_CARD)
class S4DCore(DiagonalSSMCore):
    """A diagonal S4D state-space model as a :class:`SequenceCore`."""

    def __init__(
        self,
        input_size: int,
        n_modes: int = 64,
        *,
        trainable: bool = True,
        init: Union[str, S4DLin] = "s4d-lin",
        generator: Optional[torch.Generator] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(input_size, n_modes, trainable=trainable, dtype=dtype)
        log_are, a_im, log_dt = _resolve_init(init)(
            n_modes, generator=generator, dtype=dtype
        )
        # Continuous A via the stability parametrization, seeded exactly.
        self.A = nn.Parameter(torch.complex(-torch.exp(log_are), a_im))
        P.register_parametrization(self, "A", NegExpComplex())
        with torch.no_grad():
            self.parametrizations.A.original0.copy_(log_are)
            self.parametrizations.A.original1.copy_(a_im)
        self.log_dt = nn.Parameter(log_dt)  # Δ = exp(log_dt) > 0
        self._make_B(generator, dtype)
        self._finalize()

    def _recurrence(self) -> Tuple[Tensor, Tensor]:
        A = self.A  # (N,) complex, Re < 0
        dt = torch.exp(self.log_dt)  # (N,) real > 0
        B = torch.complex(self.B_re, self.B_im)  # (N, input)
        return zoh(A, dt, B)  # (Ā, B̄)
