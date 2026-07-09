"""Render a BenchmarkResult as markdown or CSV."""

from torchdyno.benchmark.runner import BenchmarkResult


def to_markdown(result: BenchmarkResult) -> str:
    """Render a result as a markdown report (header + metric table)."""
    lines = [
        f"# Benchmark: {result.dataset}",
        "",
        f"- seeds: {result.seeds}",
        f"- wall time: {result.wall_time_s:.4f} s",
        f"- peak memory: {result.peak_memory_bytes} bytes",
        f"- config hash: {result.config_hash}",
        "",
        "| metric | mean | std |",
        "| --- | --- | --- |",
    ]
    for name, (mean, std) in result.metrics.items():
        lines.append(f"| {name} | {mean:.6g} | {std:.6g} |")
    return "\n".join(lines)


def to_csv(result: BenchmarkResult) -> str:
    """Render a result as CSV (one row per metric)."""
    lines = ["metric,mean,std"]
    for name, (mean, std) in result.metrics.items():
        lines.append(f"{name},{mean:.6g},{std:.6g}")
    return "\n".join(lines)
