"""What every learner returns, and the scoring they share."""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
    )

    from torch import Tensor

Score = TypeVar("Score")


@dataclass
class FitResult:
    """The outcome of fitting a model.

    Args:
        history: per-epoch metrics (e.g. ``{"train_loss": [...]}``); empty for
            one-shot solvers.
        best: selected hyperparameters and scores (e.g. ``{"l2": 1e-6}``).
        rng: RNG state captured at fit start, for
            :func:`~torchdyno.reproducibility.set_rng_state`.
    """

    history: dict[str, list[float]] = field(default_factory=dict)
    best: dict[str, Any] = field(default_factory=dict)
    rng: dict[str, Any] | None = None


def evaluate(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    score_fn: Callable[[Tensor, Tensor], Score],
) -> Score:
    """Score a model over a whole loader at once.

    Args:
        model: the model to score.
        loader: batches of time-first ``(x, y)``.
        score_fn: ``score_fn(prediction, target)``, returning any score.

    Returns:
        The score of the predictions for every batch, concatenated.
    """
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x))
            targets.append(y)
    # time-first (T, B, O) predictions batch on dim 1; pooled (B, C) on dim 0
    dim = 1 if preds[0].ndim == 3 else 0  # noqa: PLR2004
    return score_fn(torch.cat(preds, dim=dim), torch.cat(targets, dim=dim))


@torch.no_grad()
def feature_rows(
    model: nn.Sequential, x: Tensor, y: Tensor, washout: int = 0
) -> tuple[Tensor, Tensor]:
    """The rows a closed-form or online readout is fitted on.

    Time-aligned ``(T, B, H)`` features give one row per timestep, in time
    order, after dropping the first ``washout`` steps; pooled ``(B, H)``
    features give one row per sequence. Integer targets become one-hot.

    Args:
        model: a sequential model whose last module is the ``nn.Linear`` readout.
        x: the input batch.
        y: the targets, aligned with the features.
        washout: leading timesteps to drop.

    Returns:
        ``(n, H)`` features and ``(n, O)`` targets, in float64.

    Raises:
        TypeError: if the last module is not an ``nn.Linear``.
        ValueError: if ``washout`` is set for pooled features, a feature is not
            finite, or the targets don't match the features and the readout's
            outputs.
    """
    readout = model[-1]
    if not isinstance(readout, nn.Linear):
        raise TypeError(f"The last module must be nn.Linear, not {type(readout)}.")
    features = model[:-1](x)
    if features.ndim == 3:  # noqa: PLR2004 (T, B, H): fit every timestep
        features, y = features[washout:], y[washout:]
    elif washout:
        raise ValueError("washout needs time-aligned (T, B, H) features.")
    if not features.isfinite().all():
        rows = int((~features.isfinite()).any(-1).sum())
        raise ValueError(
            f"{rows} feature rows are not finite, e.g. NaN from positions a layer "
            "did not produce, or a state that diverged."
        )
    shape = tuple(y.shape)
    if not y.is_floating_point():
        y = F.one_hot(y, readout.out_features)
    if y.shape != (*features.shape[:-1], readout.out_features):
        raise ValueError(
            f"Targets of shape {shape} don't match features of shape "
            f"{tuple(features.shape)} and a readout with {readout.out_features} "
            "outputs: pass float targets with the outputs last, or integer class "
            "indices."
        )
    features = features.reshape(-1, features.shape[-1]).double()
    return features, y.reshape(features.shape[0], -1).double()
