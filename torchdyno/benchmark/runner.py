"""run(): fit a fresh model per seed and aggregate its metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
)

from torchdyno.learners import evaluate
from torchdyno.reproducibility import seed_all

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn

    from torchdyno.benchmark.spec import BenchmarkSpec
    from torchdyno.tasks import Task


@dataclass
class BenchmarkResult:
    """The outcome of a benchmark run.

    Args:
        dataset: the dataset name.
        metrics: ``{metric_name: (mean, std)}`` across seeds.
        per_seed: the metrics of each seed, in seed order.
        seeds: the seeds run.
        wall_time_s: total fit wall time across seeds.
    """

    dataset: str
    metrics: dict[str, tuple[float, float]]
    per_seed: list[dict[str, float]]
    seeds: list[int]
    wall_time_s: float


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)  # population variance
    return mean, var**0.5


def run(
    spec: BenchmarkSpec,
    build: Callable[[Task], tuple[nn.Module, Any]],
) -> BenchmarkResult:
    """Fit and evaluate a fresh model for every seed of ``spec``.

    Args:
        spec: the dataset and seeds.
        build: ``build(task) -> (model, learner)``, called after
            ``seed_all(seed)`` so each seed gets a freshly initialized model.
            The learner's ``fit(model, train, val)`` gets the dataset's
            validation loader, which may be ``None``.

    Returns:
        The metrics on the evaluation loader, per seed and aggregated.

    Raises:
        ValueError: if ``spec.seeds`` is empty.
    """
    if not spec.seeds:
        raise ValueError("spec.seeds must be non-empty")

    data = spec.dataset
    per_seed: list[dict[str, float]] = []
    wall_time = 0.0
    for seed in spec.seeds:
        seed_all(seed)
        model, learner = build(data.task)
        start = time.perf_counter()
        learner.fit(model, data.train_loader, data.val_loader)
        wall_time += time.perf_counter() - start
        per_seed.append(evaluate(model, data.eval_loader, data.task.metrics))

    metrics = {name: _mean_std([d[name] for d in per_seed]) for name in per_seed[0]}
    return BenchmarkResult(
        dataset=data.name,
        metrics=metrics,
        per_seed=per_seed,
        seeds=list(spec.seeds),
        wall_time_s=wall_time,
    )
