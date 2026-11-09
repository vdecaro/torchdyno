"""Sequence classification: cross-entropy loss, accuracy metric."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F

from torchdyno.metrics import Accuracy
from torchdyno.tasks.base import Task

if TYPE_CHECKING:
    from torch import Tensor


class SequenceClassification(Task):
    """One label per sequence: ``(B, C)`` logits against ``(B,)`` indices."""

    def __init__(self) -> None:
        """Initializes the task."""
        super().__init__({"accuracy": Accuracy()}, "accuracy")

    def loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Cross-entropy.

        Args:
            pred: the ``(B, C)`` logits.
            target: the ``(B,)`` class indices.

        Returns:
            The scalar loss.
        """
        return F.cross_entropy(pred, target)
