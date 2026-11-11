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
    Protocol,
    TypeVar,
)

import torch
import torch.nn.functional as F
from torch import nn

from torchdyno.matrices import Dense

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


class Learner(Protocol):
    """Anything that fits a model: what every learner here implements.

    A new learning rule needs only this method to run wherever a learner is
    expected, e.g. in :class:`Chain` or :func:`~torchdyno.benchmark.run`.
    """

    def fit(
        self,
        model: nn.Module,
        train: Iterable[tuple[Tensor, Tensor]],
        val: Iterable[tuple[Tensor, Tensor]] | None = None,
    ) -> FitResult:
        """Fit ``model`` in place.

        Args:
            model: the model to fit.
            train: batches of time-first ``(x, y)``.
            val: optional batches for selection or monitoring.

        Returns:
            What the fit recorded.
        """
        ...


class Chain:
    """Run learners one after the other on the same model.

    E.g. ``Chain(IntrinsicPlasticity(), Ridge())`` adapts a reservoir, then
    fits its readout; ``Chain(Ridge(), Backprop(...))`` fine-tunes from the
    closed-form solution.
    """

    def __init__(self, *learners: Learner):
        """Initializes the chain.

        Args:
            learners: the learners, in order.
        """
        self.learners = learners

    def fit(
        self,
        model: nn.Module,
        train: Iterable[tuple[Tensor, Tensor]],
        val: Iterable[tuple[Tensor, Tensor]] | None = None,
    ) -> FitResult:
        """Fit ``model`` with every learner in turn.

        Args:
            model: the model to fit in place.
            train: batches of time-first ``(x, y)``, given to every learner.
            val: optional batches, given to every learner.

        Returns:
            Every learner's history and selections, keyed ``"<index>.<name>"``,
            and the RNG state at the start.
        """
        chained = FitResult()
        for i, learner in enumerate(self.learners):
            result = learner.fit(model, train, val)
            chained.history |= {f"{i}.{k}": v for k, v in result.history.items()}
            chained.best |= {f"{i}.{k}": v for k, v in result.best.items()}
            chained.rng = chained.rng or result.rng
        return chained


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
    if preds[0].ndim == 3:  # noqa: PLR2004 time-aligned: rows of (T·B, O)
        pairs = zip(preds, targets, strict=True)
        rows = [_present(p.flatten(0, 1), t.flatten(0, 1)) for p, t in pairs]
        preds, targets = [r for r, _ in rows], [t for _, t in rows]
    return score_fn(torch.cat(preds), torch.cat(targets))


def find_readout(model: nn.Module, path: str | None = None) -> nn.Linear | Dense:
    """The linear readout a closed-form or online learner fits.

    Args:
        model: the model holding the readout.
        path: the readout's name in ``model``, as ``model.get_submodule`` takes
            it (e.g. ``"2"`` or ``"head.0"``); ``None`` takes the last module of
            an ``nn.Sequential``.

    Returns:
        The readout.

    Raises:
        TypeError: if the readout is not a real ``nn.Linear`` or
            :class:`~torchdyno.matrices.Dense`, or ``path`` is ``None`` for a
            model that is not an ``nn.Sequential``.
    """
    if path is None:
        if not isinstance(model, nn.Sequential):
            raise TypeError("Name the readout with readout= in a non-sequential model.")
        module = model[-1]
    else:
        module = model.get_submodule(path)
    # a complex Dense stores (out, in, 2) parts
    if not isinstance(module, nn.Linear | Dense) or module.weight.ndim != 2:  # noqa: PLR2004
        raise TypeError(
            "The readout must be a real nn.Linear or Dense, not "
            f"{type(module).__name__}."
        )
    return module


@torch.no_grad()
def feature_rows(
    model: nn.Module,
    readout: nn.Linear | Dense,
    x: Tensor,
    y: Tensor,
    washout: int = 0,
) -> tuple[Tensor, Tensor]:
    """The rows a closed-form or online readout is fitted on.

    The features are whatever reaches ``readout`` when ``model`` runs on ``x``,
    in eval mode so dropout is off. Time-aligned ``(T, B, H)`` features give one
    row per timestep, in time order, after dropping the first ``washout`` steps;
    pooled ``(B, H)`` features give one row per sequence. Integer targets become
    one-hot.

    Args:
        model: the model holding the readout.
        readout: the readout, e.g. from :func:`find_readout`.
        x: the input batch.
        y: the targets of the readout's output, aligned with the features.
        washout: leading timesteps to drop.

    Returns:
        ``(n, H)`` features and ``(n, O)`` targets, in float64.

    Raises:
        ValueError: if the readout does not run exactly once, ``washout`` is set
            for pooled features or leaves no timestep, a feature is not finite,
            or the targets don't match the features and the readout's outputs.
    """
    inputs: list[Tensor] = []
    handle = readout.register_forward_pre_hook(lambda _, args: inputs.append(args[0]))
    try:
        with restore_modes(model):
            model.eval()(x)
    finally:
        handle.remove()
    if len(inputs) != 1:
        raise ValueError(f"The readout must run once per batch, it ran {len(inputs)}.")
    features, y = drop_washout(inputs[0], y, washout)
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
    features, y = features.reshape(-1, features.shape[-1]), y.reshape(-1, y.shape[-1])
    if inputs[0].ndim == 3:  # noqa: PLR2004 time-aligned: drop the absent steps
        features, y = _present(features, y)
    if not features.isfinite().all():
        rows = int((~features.isfinite()).any(-1).sum())
        raise ValueError(
            f"{rows} feature rows are not finite, e.g. a state that diverged, or "
            "NaN in some but not all features of a step."
        )
    return features.double(), y.double()


def _present(rows: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    """Rows of a time-aligned output, and their targets, at the steps that exist."""
    keep = ~rows.isnan().all(-1)
    return rows[keep], targets[keep]
