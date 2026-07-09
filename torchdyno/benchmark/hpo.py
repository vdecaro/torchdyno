"""Grid / random hyperparameter search over the benchmark, factory-consistent."""

import itertools
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Tuple

from torchdyno.benchmark.runner import BenchmarkResult, run
from torchdyno.benchmark.spec import BenchmarkSpec


@dataclass
class SearchResult:
    """The outcome of a hyperparameter search.

    Args:
        best_config: the config with the best objective.
        best_result: the :class:`BenchmarkResult` for ``best_config``.
        best_score: the best objective value (mean across seeds).
        trials: ``(config, objective)`` for every config tried.
    """

    best_config: Dict[str, Any]
    best_result: BenchmarkResult
    best_score: float
    trials: List[Tuple[Dict[str, Any], float]]


def _grid(space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(space)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(space[k] for k in keys))]


def _sample(space: Dict[str, List[Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    keys = list(space)
    return [{k: rng.choice(space[k]) for k in keys} for _ in range(n)]


def search(
    space: Dict[str, List[Any]],
    build_from_config: Callable[[Dict[str, Any]], Callable[[Any], Tuple[Any, Any]]],
    spec: BenchmarkSpec,
    objective: str,
    mode: Literal["min", "max"] = "min",
    strategy: Literal["grid", "random"] = "grid",
    n_samples: int = 10,
    seed: int = 0,
) -> SearchResult:
    """Search ``space`` by running the benchmark per config; return the best.

    ``build_from_config(config) -> build`` maps a config point to a benchmark
    ``build(task) -> (model, learner)`` closure (the same factory ``run`` takes).
    ``objective`` is a metric name; ``mode`` says whether smaller or larger wins.
    """
    if strategy == "grid":
        configs = _grid(space)
    elif strategy == "random":
        configs = _sample(space, n_samples, seed)
    else:
        raise ValueError(f"Unknown strategy {strategy!r}. Use 'grid' or 'random'.")
    if not configs:
        raise ValueError("Empty search space.")

    trials: List[Tuple[Dict[str, Any], float]] = []
    best_config: Dict[str, Any] = {}
    best_result: BenchmarkResult = None  # type: ignore[assignment]
    best_score = float("inf") if mode == "min" else float("-inf")
    for config in configs:
        result = run(spec, build_from_config(config), config=config)
        score = result.metrics[objective][0]
        trials.append((config, score))
        better = score < best_score if mode == "min" else score > best_score
        if better:
            best_config, best_result, best_score = config, result, score
    return SearchResult(
        best_config=best_config,
        best_result=best_result,
        best_score=best_score,
        trials=trials,
    )
