"""SequenceModel: the thin composition of a core and a head."""

from typing import (
    Any,
    Optional,
    Tuple,
    Union,
)

from torch import (
    Tensor,
    nn,
)

from torchdyno.heads.base import Head
from torchdyno.models.base import (
    CoreOutput,
    SequenceCore,
)


class SequenceModel(nn.Module):
    """Compose a :class:`SequenceCore` with a :class:`Head`.

    The core produces a state sequence; the head maps it to predictions. The
    model owns no training logic — fitting is done by external machinery.
    """

    def __init__(self, core: SequenceCore, head: Head):
        super().__init__()
        self.core = core
        self.head = head

    def forward(
        self,
        x: Tensor,
        *,
        state0: Any = None,
        mask: Optional[Tensor] = None,
        return_core_output: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, CoreOutput]]:
        out = self.core(x, state0=state0, mask=mask)
        pred = self.head(out.states)
        if return_core_output:
            return pred, out
        return pred
