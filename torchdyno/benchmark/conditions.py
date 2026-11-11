"""Run one model three ways: readout, frozen and trained.

The three conditions share seeds, data and initialisation:

- ``readout``: everything before the readout frozen, the readout fitted by
  ridge regression;
- ``frozen``: the same parts frozen, a linear encoder in front and the readout
  trained by backprop;
- ``trained``: the whole model trained by backprop.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

from torch import nn

from torchdyno.benchmark.runner import run
from torchdyno.learners import (
    Backprop,
    Ridge,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Mapping,
    )

    from torch.optim import Optimizer

    from torchdyno.benchmark.runner import BenchmarkResult
    from torchdyno.benchmark.spec import BenchmarkSpec
    from torchdyno.learners import Learner
    from torchdyno.objectives import Task

CONDITIONS = ("readout", "frozen", "trained")


def run_conditions(
    spec: BenchmarkSpec,
    make_model: Callable[[], nn.Sequential],
    *,
    l2: float | list[float],
    lr: float,
    epochs: int,
    scheduler: Callable[[Optimizer], Any] | None = None,
    ridge: Mapping[str, Any] | None = None,
) -> dict[str, BenchmarkResult]:
    """Benchmark ``make_model`` under each condition on the same seeds.

    Any other condition is a :func:`~torchdyno.benchmark.run` of its own build,
    added to the result for :func:`conditions_table`, e.g.
    ``results["plasticity"] = run(spec, build)``.

    Args:
        spec: the dataset and seeds.
        make_model: builds a fresh model ending in an ``nn.Linear`` readout.
        l2: the ridge L2 value, or candidates selected on the validation loader.
        lr: the Adam learning rate for both backprop conditions.
        epochs: the epochs for both backprop conditions.
        scheduler: optional ``optimizer -> LRScheduler`` for both backprop
            conditions.
        ridge: more arguments for :class:`~torchdyno.learners.Ridge` in the
            readout condition, e.g. ``{"standardize": True}``.

    Every condition fits and scores after the dataset's washout.

    Returns:
        A result per condition, keyed by name.
    """
    # read before seeding, so peeking at a shuffled loader moves no seeded RNG
    width = next(iter(spec.dataset.train_loader))[0].shape[-1]
    washout = spec.dataset.washout

    def build(condition: str) -> Callable[[Task], tuple[nn.Module, Learner]]:
        def _build(task: Task) -> tuple[nn.Module, Learner]:
            model = make_model()
            if condition != "trained":
                model[:-1].requires_grad_(False)
            if condition == "readout":
                return model, Ridge(l2, washout, score_fn=task.primary, **(ridge or {}))
            if condition == "frozen":
                model = nn.Sequential(nn.Linear(width, width), *model)
            return model, Backprop(
                task.loss, lr=lr, epochs=epochs, scheduler=scheduler, washout=washout
            )

        return _build

    return {condition: run(spec, build(condition)) for condition in CONDITIONS}


def conditions_table(results: dict[str, BenchmarkResult]) -> str:
    """Render the results of :func:`run_conditions` as a markdown table.

    Args:
        results: the result of each condition.

    Returns:
        One row per condition: each metric as mean ± std, then the wall time.
    """
    names = list(next(iter(results.values())).metrics)
    lines = [
        "| condition | " + " | ".join(names) + " | wall time (s) |",
        "| --- |" + " --- |" * (len(names) + 1),
    ]
    for condition, result in results.items():
        cells = [f"{m:.4f} ± {s:.4f}" for m, s in result.metrics.values()]
        lines.append(
            f"| {condition} | " + " | ".join(cells) + f" | {result.wall_time_s:.1f} |"
        )
    return "\n".join(lines)
