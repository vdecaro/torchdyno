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
    from collections.abc import Callable

    from torch.optim import Optimizer

    from torchdyno.benchmark.runner import BenchmarkResult
    from torchdyno.benchmark.spec import BenchmarkSpec
    from torchdyno.tasks import Task

CONDITIONS = ("readout", "frozen", "trained")


def run_conditions(
    spec: BenchmarkSpec,
    make_model: Callable[[], nn.Sequential],
    *,
    l2: float | list[float],
    lr: float,
    epochs: int,
    scheduler: Callable[[Optimizer], Any] | None = None,
    washout: int = 0,
) -> dict[str, BenchmarkResult]:
    """Benchmark ``make_model`` under each condition on the same seeds.

    Args:
        spec: the dataset and seeds.
        make_model: builds a fresh model ending in an ``nn.Linear`` readout.
        l2: the ridge L2 value, or candidates selected on the validation loader.
        lr: the Adam learning rate for both backprop conditions.
        epochs: the epochs for both backprop conditions.
        scheduler: optional ``optimizer -> LRScheduler`` for both backprop
            conditions.
        washout: leading timesteps ridge discards.

    Returns:
        A result per condition, keyed by name.
    """
    # read before seeding, so peeking at a shuffled loader moves no seeded RNG
    width = next(iter(spec.dataset.train_loader))[0].shape[-1]

    def build(condition: str) -> Callable[[Task], tuple[nn.Module, Any]]:
        def _build(task: Task) -> tuple[nn.Module, Any]:
            model = make_model()
            if condition != "trained":
                model[:-1].requires_grad_(False)
            if condition == "readout":
                return model, Ridge(
                    l2, washout, score_fn=task.primary, mode=task.primary.mode
                )
            if condition == "frozen":
                model = nn.Sequential(nn.Linear(width, width), *model)
            return model, Backprop(task.loss, lr=lr, epochs=epochs, scheduler=scheduler)

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
