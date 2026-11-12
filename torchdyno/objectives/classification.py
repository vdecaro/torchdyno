"""Classification: accuracy and its class-balanced kin, per sequence or per step."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from torchdyno.objectives.base import (
    Metric,
    Task,
)

if TYPE_CHECKING:
    from torch import Tensor


def _confusion(pred: Tensor, target: Tensor) -> Tensor:
    """``(C, C)`` counts, the true class by row and the predicted one by column."""
    if pred.shape[:-1] != target.shape:
        raise ValueError(
            f"pred {tuple(pred.shape)} must be target {tuple(target.shape)} "
            "with a class dimension added."
        )
    classes = pred.shape[-1]
    index = target.flatten() * classes + pred.argmax(dim=-1).flatten()
    return torch.bincount(index, minlength=classes**2).reshape(classes, classes)


class Accuracy(Metric):
    """Fraction of correct top-1 predictions."""

    mode = "max"

    def __call__(self, pred: Tensor, target: Tensor) -> float:
        """Score ``pred`` against ``target``.

        Args:
            pred: the ``(..., C)`` logits.
            target: the ``(...)`` class indices.

        Returns:
            The accuracy.

        Raises:
            ValueError: unless ``pred`` has one more dimension than ``target``.
        """
        confusion = _confusion(pred, target)
        return (confusion.trace() / confusion.sum()).item()


class BalancedAccuracy(Metric):
    """The mean over classes of the fraction of each class predicted correctly.

    Classes the target does not have are left out, so on balanced data it is
    the accuracy, and a constant prediction scores ``1 / C``.
    """

    mode = "max"

    def __call__(self, pred: Tensor, target: Tensor) -> float:
        """Score ``pred`` against ``target``.

        Args:
            pred: the ``(..., C)`` logits.
            target: the ``(...)`` class indices.

        Returns:
            The mean recall over the classes in ``target``.

        Raises:
            ValueError: unless ``pred`` has one more dimension than ``target``.
        """
        confusion = _confusion(pred, target).double()
        support = confusion.sum(dim=1)
        present = support > 0
        return (confusion.diagonal()[present] / support[present]).mean().item()


class F1(Metric):
    """Macro F1: the mean over classes of ``2 TP / (2 TP + FP + FN)``.

    The mean runs over the classes that are in the target or predicted.
    """

    mode = "max"

    def __call__(self, pred: Tensor, target: Tensor) -> float:
        """Score ``pred`` against ``target``.

        Args:
            pred: the ``(..., C)`` logits.
            target: the ``(...)`` class indices.

        Returns:
            The macro-averaged F1 score.

        Raises:
            ValueError: unless ``pred`` has one more dimension than ``target``.
        """
        confusion = _confusion(pred, target).double()
        hits, predicted, actual = (
            confusion.diagonal(),
            confusion.sum(0),
            confusion.sum(1),
        )
        seen = (predicted + actual) > 0
        return (2 * hits / (predicted + actual))[seen].mean().item()


class ConfusionMatrix:
    """Counts of every true class against every predicted one.

    A score for [`evaluate()`][torchdyno.learners.evaluate], not a
    [`Metric`][torchdyno.objectives.Metric]: it is a matrix, not a number to select on.
    """

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        """Count ``pred`` against ``target``.

        Args:
            pred: the ``(..., C)`` logits.
            target: the ``(...)`` class indices.

        Returns:
            The ``(C, C)`` counts, the true class by row and the predicted one by
            column.

        Raises:
            ValueError: unless ``pred`` has one more dimension than ``target``.
        """
        return _confusion(pred, target)


class SequenceClassification(Task):
    """One label per sequence: ``(B, C)`` logits against ``(B,)`` indices."""

    def __init__(self) -> None:
        """Initializes the task."""
        super().__init__({"accuracy": Accuracy()}, "accuracy")

    def loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Cross-entropy.

        Args:
            pred: the ``(..., C)`` logits.
            target: the ``(...)`` class indices.

        Returns:
            The scalar loss.
        """
        return F.cross_entropy(pred.flatten(0, -2), target.flatten())


class TimestepClassification(SequenceClassification):
    """One label per timestep: ``(T, B, C)`` logits against ``(T, B)`` indices.

    The learners and [`evaluate()`][torchdyno.learners.evaluate] take time-aligned
    outputs as rows, so the loss and metrics are those of
    [`SequenceClassification`][torchdyno.objectives.SequenceClassification].
    """
