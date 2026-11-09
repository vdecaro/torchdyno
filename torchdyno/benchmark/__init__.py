"""Benchmarks: fit fresh models across seeds and report mean ± std."""

from .conditions import (
    CONDITIONS,
    conditions_table,
    run_conditions,
)
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
    "CONDITIONS",
    "BenchmarkDataset",
    "BenchmarkResult",
    "BenchmarkSpec",
    "conditions_table",
    "lorenz_benchmark",
    "lorenz_system",
    "run",
    "run_conditions",
    "to_markdown",
]
