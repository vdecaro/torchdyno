"""Base package of torchdyno."""

from . import data
from . import models
from . import optim
from . import training
from . import heads
from . import metrics
from . import tasks
from .model import SequenceModel
from .reproducibility import seed_all, get_rng_state, set_rng_state
from .registry import (
    ModelCard,
    create_core,
    create_head,
    create_learner,
    get_card,
    list_cores,
    list_heads,
    list_learners,
    render_catalog,
)

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torchdyno")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0+unknown"

__all__ = [
    "data",
    "models",
    "optim",
    "training",
    "heads",
    "metrics",
    "tasks",
    "SequenceModel",
    "seed_all",
    "get_rng_state",
    "set_rng_state",
    "ModelCard",
    "create_core",
    "create_learner",
    "create_head",
    "list_cores",
    "list_learners",
    "list_heads",
    "get_card",
    "render_catalog",
]
