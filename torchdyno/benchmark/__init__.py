"""Benchmarks: fit fresh models across seeds and report mean ± std."""

from .dataset import BenchmarkDataset
from .datasets import (
    lorenz_benchmark,
    lorenz_system,
)
from .report import to_markdown
from .runner import (
    BenchmarkResult,
    run,
)
from .spec import BenchmarkSpec

__all__ = [
    "BenchmarkDataset",
    "BenchmarkResult",
    "BenchmarkSpec",
    "lorenz_benchmark",
    "lorenz_system",
    "run",
    "to_markdown",
]
