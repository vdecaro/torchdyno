"""Deterministic seeding and RNG-state capture across python/numpy/torch.

``seed_all`` makes reservoir initialization and training reproducible — a
non-negotiable for a library whose models are randomly initialized. CUDA
adds caveats: some ops have no deterministic implementation, and
``CUBLAS_WORKSPACE_CONFIG`` may be required; ``deterministic=True`` sets the
flags in ``warn_only`` mode so CPU runs are unaffected.
"""

import random
from typing import Any, Dict

import numpy as np
import torch


def seed_all(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy, and torch (+ CUDA), optionally forcing determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rng_state() -> Dict[str, Any]:
    """Capture the RNG state of python, numpy, and torch (+ CUDA if present)."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore the RNG state captured by :func:`get_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
