"""Base package of torchdyno."""

from . import data
from . import models
from . import optim

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torchdyno")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0+unknown"

__all__ = ["data", "models", "optim"]
