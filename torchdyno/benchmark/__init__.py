"""Benchmark harness: run a model+learner across seeds, report mean±std."""

from .dataset import BenchmarkDataset
from .spec import BenchmarkSpec
from .runner import BenchmarkResult, run

__all__ = ["BenchmarkDataset", "BenchmarkSpec", "BenchmarkResult", "run"]
