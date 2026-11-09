"""Deterministic seeding and RNG-state capture across Python and torch.

``seed_all`` makes initialization and training reproducible. CUDA adds caveats:
some ops have no deterministic implementation, and ``CUBLAS_WORKSPACE_CONFIG``
may be required; ``deterministic=True`` sets the flags in ``warn_only`` mode so
CPU runs are unaffected.
"""

import random
from typing import Any

import torch


def seed_all(seed: int, deterministic: bool = True) -> None:
    """Seed Python and torch on every device, optionally forcing determinism.

    Args:
        seed: the seed.
        deterministic: whether to ask torch for deterministic algorithms.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rng_state() -> dict[str, Any]:
    """Capture the RNG state of Python and torch (+ CUDA if present).

    Returns:
        The state, for :func:`set_rng_state`.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore the RNG state captured by :func:`get_rng_state`.

    Args:
        state: the captured state.
    """
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
