"""Name → builder registry for cores, learners, and heads, with model cards.

Registration is via decorators applied at import time (timm/HF style);
``import torchdyno`` populates the registry. Lookups read the dicts at call
time. This module imports only the standard library, so it never forms an
import cycle with the packages that register into it.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Tuple


@dataclass(frozen=True)
class ModelCard:
    """Discoverability metadata for a registered core.

    Args:
        name: registry name (must match the ``register_core`` name).
        family: model family, e.g. ``"reservoir"`` or ``"assembly"``.
        paper: human citation string (authors, title, venue, year).
        description: 1-3 sentence summary.
        admits: registered ``Learner`` names the core is trained by (``.fit``).
        adapters: ``CoreAdapter`` names (documented; e.g. ``("ip",)``).
        tasks: natural task-shape strings (the Task layer is not built yet).
        default_config: a small, runnable base config for ``create_core``.
    """

    name: str
    family: str
    paper: str
    description: str
    admits: Tuple[str, ...]
    adapters: Tuple[str, ...]
    tasks: Tuple[str, ...]
    default_config: Mapping[str, Any]


@dataclass
class _CoreReg:
    builder: Callable[..., Any]
    card: ModelCard


_CORES: Dict[str, _CoreReg] = {}
_LEARNERS: Dict[str, Callable[..., Any]] = {}
_HEADS: Dict[str, Callable[..., Any]] = {}


def register_core(name: str, *, card: ModelCard) -> Callable[[Any], Any]:
    """Decorator: register a core class under ``name`` with its ``card``."""

    def deco(cls):
        if card.name != name:
            raise ValueError(f"Card name {card.name!r} != registered name {name!r}.")
        if name in _CORES and _CORES[name].builder is not cls:
            raise ValueError(f"Core {name!r} already registered to a different class.")
        _CORES[name] = _CoreReg(builder=cls, card=card)
        return cls

    return deco


def register_learner(name: str) -> Callable[[Any], Any]:
    """Decorator: register a Learner class under ``name``."""

    def deco(cls):
        if name in _LEARNERS and _LEARNERS[name] is not cls:
            raise ValueError(f"Learner {name!r} already registered.")
        _LEARNERS[name] = cls
        return cls

    return deco


def register_head(name: str) -> Callable[[Any], Any]:
    """Decorator: register a Head class under ``name``."""

    def deco(cls):
        if name in _HEADS and _HEADS[name] is not cls:
            raise ValueError(f"Head {name!r} already registered.")
        _HEADS[name] = cls
        return cls

    return deco


def create_core(name: str, **overrides) -> Any:
    """Build a registered core, merging its card ``default_config`` with overrides."""
    if name not in _CORES:
        raise KeyError(f"Unknown core {name!r}. Available: {sorted(_CORES)}")
    reg = _CORES[name]
    cfg = {**reg.card.default_config, **overrides}
    return reg.builder(**cfg)


def create_learner(name: str, **cfg) -> Any:
    """Build a registered learner by name."""
    if name not in _LEARNERS:
        raise KeyError(f"Unknown learner {name!r}. Available: {sorted(_LEARNERS)}")
    return _LEARNERS[name](**cfg)


def create_head(name: str, **cfg) -> Any:
    """Build a registered head by name."""
    if name not in _HEADS:
        raise KeyError(f"Unknown head {name!r}. Available: {sorted(_HEADS)}")
    return _HEADS[name](**cfg)


def list_cores() -> List[str]:
    """Sorted names of registered cores."""
    return sorted(_CORES)


def list_learners() -> List[str]:
    """Sorted names of registered learners."""
    return sorted(_LEARNERS)


def list_heads() -> List[str]:
    """Sorted names of registered heads."""
    return sorted(_HEADS)


def get_card(name: str) -> ModelCard:
    """Return the ModelCard for a registered core."""
    if name not in _CORES:
        raise KeyError(f"Unknown core {name!r}. Available: {sorted(_CORES)}")
    return _CORES[name].card


def render_catalog() -> str:
    """Render the registered cores as a markdown catalog string."""
    lines = ["# TorchDyno model catalog", ""]
    for name in list_cores():
        card = _CORES[name].card
        caps = create_core(name).capabilities
        adapters = f" | adapters: {', '.join(card.adapters)}" if card.adapters else ""
        lines += [
            f"## {card.name}  ({card.family})",
            "",
            card.description,
            "",
            f"- **Paper:** {card.paper}",
            f"- **Trains with:** {', '.join(card.admits) or '—'}{adapters}",
            f"- **Tasks:** {', '.join(card.tasks)}",
            (
                f"- **Capabilities:** compute_mode={caps.compute_mode}, "
                f"differentiable={caps.differentiable}, supports_step={caps.supports_step}, "
                f"trainable_recurrence={caps.trainable_recurrence}"
            ),
            "",
        ]
    return "\n".join(lines)
