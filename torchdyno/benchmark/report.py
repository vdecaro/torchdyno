"""Render a BenchmarkResult as markdown."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchdyno.benchmark.runner import BenchmarkResult


def to_markdown(result: BenchmarkResult) -> str:
    """Render a result as a markdown report.

    Args:
        result: the result to render.

    Returns:
        A header with the seeds and wall time, then a metric table.
    """
    lines = [
        f"# Benchmark: {result.dataset}",
        "",
        f"- seeds: {result.seeds}",
        f"- wall time: {result.wall_time_s:.4f} s",
        "",
        "| metric | mean | std |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {name} | {mean:.6g} | {std:.6g} |"
        for name, (mean, std) in result.metrics.items()
    ]
    return "\n".join(lines)
