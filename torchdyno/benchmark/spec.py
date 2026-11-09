"""BenchmarkSpec: a dataset and the seeds to average over."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchdyno.benchmark.dataset import BenchmarkDataset


@dataclass
class BenchmarkSpec:
    """What to benchmark: a :class:`BenchmarkDataset` and the seeds to run.

    Args:
        dataset: the data and task.
        seeds: the seeds; each gets a freshly initialized model.
    """

    dataset: BenchmarkDataset
    seeds: list[int]
