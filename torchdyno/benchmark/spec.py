"""BenchmarkSpec: a dataset + the seeds to average over."""

from dataclasses import dataclass
from typing import List

from torchdyno.benchmark.dataset import BenchmarkDataset


@dataclass
class BenchmarkSpec:
    """What to benchmark: a :class:`BenchmarkDataset` and the seeds to run."""

    dataset: BenchmarkDataset
    seeds: List[int]
