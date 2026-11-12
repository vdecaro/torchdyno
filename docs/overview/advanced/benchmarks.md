# Benchmarks

[`run()`][torchdyno.benchmark.run] fits a fresh model for every seed of a
[`BenchmarkSpec`][torchdyno.benchmark.BenchmarkSpec] and reports each metric of the
dataset's task as a mean and a standard deviation, as
[Benchmark across seeds](../getting-started/benchmark-across-seeds.md) shows.

## Datasets

A [`BenchmarkDataset`][torchdyno.benchmark.BenchmarkDataset] holds the train, evaluation
and optional validation loaders, the task, and the washout. Any iterable of
`(x, y)` batches is a loader. These are built in:

| Dataset | Task |
|---|---|
| [`lorenz_benchmark()`][torchdyno.benchmark.lorenz_benchmark] | forecast the Lorenz system |
| [`mackey_glass_benchmark()`][torchdyno.benchmark.mackey_glass_benchmark] | forecast the Mackey-Glass delay equation |
| [`narma_benchmark()`][torchdyno.benchmark.narma_benchmark] | predict the output of a NARMA system from its random input |
| [`memory_capacity_benchmark()`][torchdyno.benchmark.memory_capacity_benchmark] | recall the input of each of the last steps, scored by memory capacity |
| [`ucr_benchmark()`][torchdyno.benchmark.ucr_benchmark] | classify a UCR or UEA dataset read from local files |

The synthetic ones are cut into contiguous train, validation and evaluation
segments. [`ucr_benchmark()`][torchdyno.benchmark.ucr_benchmark] reads the archives'
`.tsv` and `.ts` files, given the folder the archive was extracted to:

```python
from torchdyno.benchmark import ucr_benchmark

data = ucr_benchmark("~/data/Univariate_arff", "GunPoint", batch_size=32, val=0.2)
```

## Readout, frozen, trained

[`run_conditions()`][torchdyno.benchmark.run_conditions] runs one model three ways on the
same seeds, data and initialisation:

- `readout`: everything before the readout frozen, the readout fitted by
    ridge regression;
- `frozen`: the same parts frozen, a linear encoder in front and the readout
    trained by backprop;
- `trained`: the whole model trained by backprop.

[`conditions_table()`][torchdyno.benchmark.conditions_table] lays the results side by side, as
in the [conditions example](../examples/conditions.md).

## Reproducibility

[`seed_all()`][torchdyno.seed_all] seeds Python, torch and any generators given, such
as a `DataLoader`'s, and by default asks torch for deterministic algorithms;
[`run()`][torchdyno.benchmark.run] calls it before every build. Every
[`FitResult`][torchdyno.learners.FitResult] keeps the RNG state its fit started
from, which [`set_rng_state()`][torchdyno.set_rng_state] restores to replay the fit.
