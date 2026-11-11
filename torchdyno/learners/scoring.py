"""Scoring a model over a loader, and the washout and modes scoring respects."""

from __future__ import annotations

from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    TypeVar,
)

import torch

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
        Iterator,
    )

    from torch import (
        Tensor,
        nn,
    )


Score = TypeVar("Score")


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
        batches of different lengths combine; steps a sequence does not have,
        all NaN, are left out.

    Raises:
        ValueError: if ``loader`` has no batches.
    """
    preds, targets = [], []
    with torch.no_grad(), restore_modes(model):
        model.eval()
        for x, y in loader:
            pred, target = drop_washout(model(x), y, washout)
            preds.append(pred)
            targets.append(target)
    if not preds:
        raise ValueError("The loader has no batches to score.")
    return score_outputs(preds, targets, score_fn)


def score_outputs(
    preds: list[Tensor],
    targets: list[Tensor],
    score_fn: Callable[[Tensor, Tensor], Score],
) -> Score:
    """Score the outputs of several batches together, as :func:`evaluate` does.

    Args:
        preds: every batch's predictions, after the washout.
        targets: every batch's targets, aligned with ``preds``.
        score_fn: ``score_fn(prediction, target)``, returning any score.

    Returns:
        The score of the predictions, concatenated; time-aligned ones as rows,
        without the steps a sequence does not have.
    """
    if preds[0].ndim == 3:  # noqa: PLR2004 time-aligned: rows of (T·B, O)
        pairs = zip(preds, targets, strict=True)
        rows = [_present(p.flatten(0, 1), t.flatten(0, 1)) for p, t in pairs]
        preds, targets = [r for r, _ in rows], [t for _, t in rows]
    return score_fn(torch.cat(preds), torch.cat(targets))


def _present(rows: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    """Rows of a time-aligned output, and their targets, at the steps that exist."""
    keep = ~rows.isnan().all(-1)
    return rows[keep], targets[keep]
