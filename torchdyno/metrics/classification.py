"""Classification metrics (higher is better). Tolerant of one-hot or index
targets, and of sequence-level ``(B, C)`` or dense ``(T, B, C)`` shapes."""

import torch
from torch import Tensor

from torchdyno.metrics.base import Metric


def to_indices(pred: Tensor, target: Tensor) -> Tensor:
    """Return class indices for ``target``.

    If ``target`` has the same shape as ``pred`` (one-hot logits/probabilities),
    argmax the last dim; otherwise cast the existing indices to long.
    """
    if target.shape == pred.shape:
        target = target.argmax(dim=-1)
    return target.long()


class Accuracy(Metric):
    """Fraction of correct top-1 predictions over all leading dims."""

    name = "accuracy"
    mode = "max"

    def __call__(self, pred: Tensor, target: Tensor) -> float:
        preds = pred.argmax(dim=-1)
        idx = to_indices(pred, target)
        return (preds == idx).float().mean().item()


class MacroF1(Metric):
    """Unweighted mean of per-class F1 over the classes present in pred ∪ target."""

    name = "macro_f1"
    mode = "max"

    def __call__(self, pred: Tensor, target: Tensor) -> float:
        preds = pred.argmax(dim=-1).flatten()
        tgt = to_indices(pred, target).flatten()
        classes = torch.unique(torch.cat([preds, tgt]))
        f1s = []
        for c in classes:
            tp = ((preds == c) & (tgt == c)).sum().float()
            fp = ((preds == c) & (tgt != c)).sum().float()
            fn = ((preds != c) & (tgt == c)).sum().float()
            denom = 2 * tp + fp + fn
            f1s.append((2 * tp / denom).item() if denom > 0 else 0.0)
        return sum(f1s) / len(f1s) if f1s else 0.0
