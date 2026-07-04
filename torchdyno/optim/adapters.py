"""CoreAdapter: base for algorithms that modify a recurrent unit in place.

A core-modifying adapter attaches to a recurrent unit from the outside, wraps
its forward computation and/or injects parameters, and updates them by a custom
(often non-autograd) rule. Lifecycle: ``compile`` (attach) → repeated ``step``
(update) → ``detach`` (restore). Intrinsic Plasticity is the reference case.
"""

from abc import (
    ABC,
    abstractmethod,
)
from typing import Any


class CoreAdapter(ABC):
    """External algorithm that adapts a recurrent unit's own parameters."""

    @abstractmethod
    def compile(self, target: Any) -> None:
        """Attach to ``target`` (e.g. register a state hook, prepare state)."""

    @abstractmethod
    def step(self) -> None:
        """Apply one update to the adapted parameters."""

    @abstractmethod
    def detach(self) -> None:
        """Restore ``target`` to its unadapted state."""
