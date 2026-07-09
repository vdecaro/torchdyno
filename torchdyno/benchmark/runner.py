"""run(): loop seeds over fresh model+learner, aggregate metric mean±std."""

import hashlib
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from torchdyno.benchmark.spec import BenchmarkSpec
from torchdyno.reproducibility import seed_all


@dataclass
class BenchmarkResult:
    """The outcome of a benchmark run.

    Args:
        dataset: the dataset name.
        metrics: ``{metric_name: (mean, std)}`` across seeds.
        per_seed: the raw per-seed metric dicts.
        seeds: the seeds run.
        wall_time_s: total fit wall-time across seeds.
        peak_memory_bytes: best-effort peak memory (CUDA allocator if present,
            else a ``tracemalloc`` Python-heap peak — a CPU proxy).
        config_hash: a stable replay key for this run.
    """

    dataset: str
    metrics: Dict[str, Tuple[float, float]]
    per_seed: List[Dict[str, float]]
    seeds: List[int]
    wall_time_s: float
    peak_memory_bytes: Optional[int]
    config_hash: str


def _mean_std(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n  # population variance
    return mean, var ** 0.5


def _evaluate(model: Any, loader: Any, task: Any) -> Dict[str, float]:
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x))
            targets.append(y)
    dim = 1 if preds[0].ndim == 3 else 0
    pred = torch.cat(preds, dim=dim)
    target = torch.cat(targets, dim=dim)
    return task.metrics(pred, target)


def _config_hash(spec: BenchmarkSpec, config: Optional[Dict[str, Any]]) -> str:
    key = repr(
        (spec.dataset.name, list(spec.seeds), type(spec.dataset.task).__name__, config)
    )
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def run(
    spec: BenchmarkSpec,
    build: Callable[[Any], Tuple[Any, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> BenchmarkResult:
    """Benchmark ``build`` on ``spec`` across seeds; return aggregated metrics.

    ``build(task) -> (model, learner)`` is called fresh after ``seed_all(seed)``
    each run, so each seed gets a freshly-initialized model.

    ``config_hash`` covers the dataset name, seeds, task type, and ``config``
    only — not the ``build`` closure. Pass a distinguishing ``config`` dict to
    tell apart different models on the same dataset and seeds.
    """
    if not spec.seeds:
        raise ValueError("spec.seeds must be non-empty")

    task = spec.dataset.task
    cuda = torch.cuda.is_available()
    started_trace = False
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    elif not tracemalloc.is_tracing():
        tracemalloc.start()
        started_trace = True

    per_seed: List[Dict[str, float]] = []
    wall_time = 0.0
    try:
        for seed in spec.seeds:
            seed_all(seed)
            model, learner = build(task)
            t0 = time.perf_counter()
            learner.fit(model, spec.dataset.train_loader)
            wall_time += time.perf_counter() - t0
            per_seed.append(_evaluate(model, spec.dataset.eval_loader, task))

        if cuda:
            peak = torch.cuda.max_memory_allocated()
        else:
            peak = tracemalloc.get_traced_memory()[1]
    finally:
        if started_trace:
            tracemalloc.stop()

    names = per_seed[0].keys()
    metrics = {name: _mean_std([d[name] for d in per_seed]) for name in names}
    return BenchmarkResult(
        dataset=spec.dataset.name,
        metrics=metrics,
        per_seed=per_seed,
        seeds=list(spec.seeds),
        wall_time_s=wall_time,
        peak_memory_bytes=int(peak),
        config_hash=_config_hash(spec, config),
    )
