"""Benchmarks: fit fresh models across seeds and report mean ± std."""

from .dataset import BenchmarkDataset
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
    "run",
    "to_markdown",
]
