# Benchmark across seeds

One run says little about a randomly initialised model.
[`run()`][torchdyno.benchmark.run] fits a fresh model for every seed and reports
each metric of the task as a mean and a standard deviation.

## A dataset and a builder

A [`BenchmarkDataset`][torchdyno.benchmark.BenchmarkDataset] holds the loaders, the
[`Task`][torchdyno.objectives.Task] they pose and the washout. The builder takes
the task and returns a fresh model and its learner; `run` seeds every
generator before each call, so each seed draws its own model.

```python
from torch import nn

from torchdyno.benchmark import BenchmarkSpec, lorenz_benchmark, run, to_markdown
from torchdyno.learners import Ridge
from torchdyno.matrices import Dense, spectral, uniform
from torchdyno.sequence import LeakyCell, Recurrent

data = lorenz_benchmark(length=2000, washout=100)

def build(task):
    reservoir = Recurrent(
        LeakyCell(
            input=Dense(3, 200, init=uniform(1.0)),
            recurrent=Dense(200, 200, init=spectral(uniform(1.0), 0.9)),
            leak=0.5,
        )
    ).requires_grad_(False)
    model = nn.Sequential(reservoir, nn.Linear(200, 3))
    return model, Ridge([1e-8, 1e-6, 1e-4], washout=100, score_fn=task.primary)
```

## Run and report

```python
result = run(BenchmarkSpec(data, seeds=[0, 1, 2]), build)
print(result.metrics)  # {"nrmse": (mean, std), "mse": (mean, std)}
print(to_markdown(result))
```

`result.per_seed` keeps each seed's metrics and `result.fits` what each fit
returned, such as the penalty Ridge chose.

## Where to go next

[`run_conditions()`][torchdyno.benchmark.run_conditions] runs the same model three ways on
the same seeds: readout only, frozen, and fully trained. See
[Benchmarks](../advanced/benchmarks.md) for the conditions and the other datasets.
