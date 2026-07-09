"""Benchmark harness: run a model+learner across seeds, report mean±std."""

from .dataset import BenchmarkDataset
from .spec import BenchmarkSpec
from .runner import BenchmarkResult, run
from .report import to_markdown, to_csv

__all__ = [
    "BenchmarkDataset",
    "BenchmarkSpec",
    "BenchmarkResult",
    "run",
    "to_markdown",
    "to_csv",
]
