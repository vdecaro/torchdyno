"""The input checks and the missing-step mask every sequence layer shares."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from torch import Tensor


def _absent(x: Tensor) -> Tensor:
    """The ``(T, B)`` steps a sequence does not have: NaN in every feature."""
    return x.isnan().all(-1)


def _check(x: Tensor, h0: Tensor | None = None, state_size: int = 0) -> None:
    """Raise unless ``x`` is a ``(T, B, F)`` sequence and ``h0`` a ``(B, H)`` state."""
    if x.ndim != 3 or x.shape[0] == 0:  # noqa: PLR2004
        raise ValueError(
            f"Expected a time-first (T, B, F) sequence with T >= 1, got shape "
            f"{tuple(x.shape)}; add a batch dimension with x.unsqueeze(1)."
        )
    if h0 is not None and h0.shape != (x.shape[1], state_size):
        raise ValueError(
            f"h0 must be (B, state_size) = {(x.shape[1], state_size)}, got "
            f"{tuple(h0.shape)}."
        )
