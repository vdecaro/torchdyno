"""Package collecting all datasets available and useful for experimentation.

Datasets are imported lazily so that ``import torchdyno`` never pulls optional
dependencies (``torchvision`` for SequentialMNIST, ``pandas`` for HHAR). Those
live in the ``datasets`` extra: ``pip install 'torchdyno[datasets]'``.
"""

from typing import TYPE_CHECKING

# name -> (submodule, required optional dependency or None, extra name or None)
_LAZY = {
    "SequentialMNIST": (".seq_mnist", "torchvision", "datasets"),
    "WESADDataset": (".wesad", None, None),
    "HHARDataset": (".hhar", "pandas", "datasets"),
    "LorenzSystem": (".lorenz_system", None, None),
    "MemoryCapacityDataset": (".memory_capacity", None, None),
}

__all__ = list(_LAZY)

if TYPE_CHECKING:  # for type checkers / IDEs only — not executed at runtime
    from .hhar import HHARDataset
    from .lorenz_system import LorenzSystem
    from .memory_capacity import MemoryCapacityDataset
    from .seq_mnist import SequentialMNIST
    from .wesad import WESADDataset


def __getattr__(name: str):
    import importlib

    try:
        module_path, dep, extra = _LAZY[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc
    try:
        module = importlib.import_module(module_path, __name__)
    except ImportError as exc:
        if dep is not None:
            raise ImportError(
                f"{name} requires the optional dependency {dep!r}. "
                f"Install it with: pip install 'torchdyno[{extra}]'"
            ) from exc
        raise
    return getattr(module, name)


def __dir__():
    return sorted(__all__)
