"""Finding a model's linear readout, the rows it is fitted on, and caching them."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

import torch
import torch.nn.functional as F
from torch import nn

from torchdyno.learners.scoring import (
    _present,
    drop_washout,
    restore_modes,
)
from torchdyno.matrices.dense import Dense

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor


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


@torch.no_grad()
def cache_features(
    prefix: nn.Module, loader: Iterable[tuple[Tensor, Tensor]]
) -> list[tuple[Tensor, Tensor]]:
    """Run a frozen part of a model once over a loader, in eval mode.

    The result is a loader for the rest of the model, so a readout can be
    fitted and scored many times without rerunning the part before it, e.g.
    ``Ridge().fit(nn.Sequential(model[-1]), cache_features(model[:-1], train))``,
    which fits ``model[-1]`` in place.

    Args:
        prefix: the part of the model to run.
        loader: batches of time-first ``(x, y)``.

    Returns:
        The batches ``(prefix(x), y)``.
    """
    with restore_modes(prefix):
        prefix.eval()
        return [(prefix(x), y) for x, y in loader]
