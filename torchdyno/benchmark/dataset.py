"""BenchmarkDataset: fixed loaders + the declared task for a benchmark."""

from dataclasses import dataclass
from typing import Any, Iterable

from torchdyno.tasks import Task


@dataclass
class BenchmarkDataset:
    """A benchmark's data: fixed (built-once) loaders and its declared task.

    ``train_loader`` and ``eval_loader`` are any iterable of ``(x, y)`` batches
    in time-first ``(T, B, F)`` layout (a ``list[(x, y)]`` for the single-batch
    synthetic cases; a ``DataLoader`` otherwise). Built once and reused across
    seeds — the data is held fixed while the model varies.
    """

    name: str
    task: Task
    train_loader: Iterable[Any]
    eval_loader: Iterable[Any]
