"""BenchmarkDataset: fixed loaders and the declared task for a benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor

    from torchdyno.objectives import Task


@dataclass
class BenchmarkDataset:
    """A benchmark's data: loaders built once and reused across seeds.

    Each loader is any iterable of time-first ``(x, y)`` batches: a list for
    single-batch synthetic data, a ``DataLoader`` otherwise.

    Args:
        name: the dataset name.
        task: the task the data poses.
        train_loader: the training batches.
        eval_loader: the batches every reported metric is computed on.
        val_loader: batches for selecting hyperparameters, or ``None``.
        washout: leading timesteps of time-aligned predictions left unscored,
            e.g. a reservoir's warm-up.
    """

    name: str
    task: Task
    train_loader: Iterable[tuple[Tensor, Tensor]]
    eval_loader: Iterable[tuple[Tensor, Tensor]]
    val_loader: Iterable[tuple[Tensor, Tensor]] | None = None
    washout: int = 0
