"""Classification metrics (higher is better)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchdyno.metrics.base import Metric

if TYPE_CHECKING:
    from torch import Tensor


class Accuracy(Metric):
    """Fraction of correct top-1 predictions."""

    mode = "max"

    def __call__(self, pred: Tensor, target: Tensor) -> float:
        """Score ``pred`` against ``target``.

        Args:
            pred: the ``(B, C)`` logits.
            target: the ``(B,)`` class indices.

        Returns:
            The accuracy.

        Raises:
            ValueError: unless ``pred`` has one more dimension than ``target``.
        """
        if pred.shape[:-1] != target.shape:
            raise ValueError(
                f"pred {tuple(pred.shape)} must be target {tuple(target.shape)} "
                "with a class dimension added."
            )
        return (pred.argmax(dim=-1) == target).float().mean().item()
