"""Benchmark harness: run a model+learner across seeds, report mean±std."""

from .dataset import BenchmarkDataset
from .spec import BenchmarkSpec
from .runner import BenchmarkResult, run
from .report import to_markdown, to_csv
from .datasets import (
    memory_capacity_benchmark,
    lorenz_benchmark,
    sequential_mnist_benchmark,
)

__all__ = [
    "BenchmarkDataset",
    "BenchmarkSpec",
    "BenchmarkResult",
    "run",
    "to_markdown",
    "to_csv",
    "memory_capacity_benchmark",
    "lorenz_benchmark",
    "sequential_mnist_benchmark",
]
