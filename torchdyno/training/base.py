"""The uniform training surface: a Learner produces a FitResult from a model.

Training algorithms are external objects that attach to a
:class:`~torchdyno.model.SequenceModel` and set its parameters — the model
itself owns no training logic.
"""

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Dict,
    Optional,
    Protocol,
    runtime_checkable,
)


@dataclass
class FitResult:
    """The outcome of fitting a model.

    Args:
        history: per-step/epoch metrics (e.g. ``{"train_loss": [...]}``);
            empty for one-shot solvers.
        best: selected hyperparameters and scores (e.g. ``{"l2": 1e-6}``).
        extras: learner-specific artifacts (e.g. ridge ``A``/``B`` matrices).
        rng: RNG state captured at fit start (for reproducing the run).
    """

    history: Dict[str, Any] = field(default_factory=dict)
    best: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)
    rng: Optional[Dict[str, Any]] = None


@runtime_checkable
class Learner(Protocol):
    """Anything that can fit a model: ``fit(model, train, val=None) -> FitResult``."""

    def fit(self, model: Any, train: Any, val: Optional[Any] = None) -> FitResult:
        ...
