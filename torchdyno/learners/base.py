"""What every learner returns, and the scoring they share."""

from __future__ import annotations

from contextlib import contextmanager
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
        Iterator,
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


@contextmanager
def restore_modes(model: nn.Module) -> Iterator[None]:
    """Restore the train/eval mode of every module of ``model`` on exit.

    Args:
        model: the model whose modes to restore.

    Yields:
        Nothing; switch modes freely inside the block.
    """
    modes = [(module, module.training) for module in model.modules()]
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


def drop_washout(out: Tensor, y: Tensor, washout: int) -> tuple[Tensor, Tensor]:
    """Drop the first ``washout`` timesteps of time-aligned outputs and targets.

    Args:
        out: ``(T, B, ·)`` time-aligned or ``(B, ·)`` pooled outputs.
        y: the targets, aligned with ``out``.
        washout: leading timesteps to drop.

    Returns:
        ``out`` and ``y`` from timestep ``washout`` on; pooled ones unchanged.

    Raises:
        ValueError: if ``washout`` is set for pooled outputs, or leaves no
            timestep.
    """
    if out.ndim != 3:  # noqa: PLR2004 pooled (B, ·)
        if washout:
            raise ValueError("washout needs time-aligned (T, B, ·) outputs.")
        return out, y
    if not 0 <= washout < out.shape[0]:
        raise ValueError(
            f"washout must be in [0, {out.shape[0]}) for {out.shape[0]} timesteps, "
            f"got {washout}."
        )
    return out[washout:], y[washout:]


def evaluate(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    score_fn: Callable[[Tensor, Tensor], Score],
    washout: int = 0,
) -> Score:
    """Score a model over a whole loader at once.

    Args:
        model: the model to score.
        loader: batches of time-first ``(x, y)``.
        score_fn: ``score_fn(prediction, target)``, returning any score.
        washout: leading timesteps of time-aligned predictions left unscored,
            as for the learners that fit after a washout.

    Returns:
        The score of the predictions for every batch, concatenated. Time-aligned
        ``(T, B, O)`` predictions are scored as one ``(O,)`` row per timestep, so
        batches of different lengths combine.
    """
    preds, targets = [], []
    with torch.no_grad(), restore_modes(model):
        model.eval()
        for x, y in loader:
            pred, target = drop_washout(model(x), y, washout)
            preds.append(pred)
            targets.append(target)
    if preds[0].ndim == 3:  # noqa: PLR2004 time-aligned: rows of (T·B, O)
        preds = [pred.flatten(0, 1) for pred in preds]
        targets = [target.flatten(0, 1) for target in targets]
    return score_fn(torch.cat(preds), torch.cat(targets))


@torch.no_grad()
def feature_rows(
    model: nn.Sequential, x: Tensor, y: Tensor, washout: int = 0
) -> tuple[Tensor, Tensor]:
    """The rows a closed-form or online readout is fitted on.

    Time-aligned ``(T, B, H)`` features give one row per timestep, in time
    order, after dropping the first ``washout`` steps; pooled ``(B, H)``
    features give one row per sequence. Integer targets become one-hot. The
    features are computed in eval mode, so dropout is off.

    Args:
        model: a sequential model whose last module is the ``nn.Linear`` readout.
        x: the input batch.
        y: the targets, aligned with the features.
        washout: leading timesteps to drop.

    Returns:
        ``(n, H)`` features and ``(n, O)`` targets, in float64.

    Raises:
        TypeError: if the last module is not an ``nn.Linear``.
        ValueError: if ``washout`` is set for pooled features or leaves no
            timestep, a feature is not finite, or the targets don't match the
            features and the readout's outputs.
    """
    readout = model[-1]
    if not isinstance(readout, nn.Linear):
        raise TypeError(f"The last module must be nn.Linear, not {type(readout)}.")
    with restore_modes(model):
        features = model[:-1].eval()(x)
    features, y = drop_washout(features, y, washout)
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
