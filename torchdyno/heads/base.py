"""Head base class: maps a core's state sequence to predictions."""

from torch import (
    Tensor,
    nn,
)


class Head(nn.Module):
    """Base class for prediction heads.

    A head is an ordinary trainable ``nn.Module`` that also supports having its
    weights set directly (``set_weight``) so that closed-form solvers (e.g.
    ridge regression) can write the solution without gradient descent.
    """

    def set_weight(self, weight: Tensor) -> None:
        """Set the head's weight in place (for closed-form fitting)."""
        raise NotImplementedError
