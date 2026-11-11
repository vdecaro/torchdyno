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
    mackey_glass,
    mackey_glass_benchmark,
    memory_capacity_benchmark,
    narma,
    narma_benchmark,
)
from .report import to_markdown
from .runner import (
    BenchmarkResult,
    run,
)
from .spec import BenchmarkSpec
from .ucr import (
    load_ucr,
    ucr_benchmark,
)

__all__ = [
    "CONDITIONS",
    "BenchmarkDataset",
    "BenchmarkResult",
    "BenchmarkSpec",
    "conditions_table",
    "load_ucr",
    "lorenz_benchmark",
    "lorenz_system",
    "mackey_glass",
    "mackey_glass_benchmark",
    "memory_capacity_benchmark",
    "narma",
    "narma_benchmark",
    "run",
    "run_conditions",
    "to_markdown",
    "ucr_benchmark",
]
