# Readout, frozen, trained

One AdaDiag model run three ways on the same seeds: only the readout fitted by ridge; the same parts frozen, with a linear encoder in front and the readout trained by backprop; and everything trained. The model is the [AdaDiag](adadiag.md) model, built by a function so each condition and each seed gets a fresh copy, and [`run_conditions()`][torchdyno.benchmark.run_conditions] runs the comparison.

Use it when you want to understand:

- [`run_conditions()`][torchdyno.benchmark.run_conditions] and [`conditions_table()`][torchdyno.benchmark.conditions_table],
- how a [`BenchmarkDataset`][torchdyno.benchmark.BenchmarkDataset] and a [`BenchmarkSpec`][torchdyno.benchmark.BenchmarkSpec] describe a benchmark,
- how to read a result as a mean and a standard deviation across seeds.

Run it from the repository root with:

```bash
uv run python -m examples.conditions
```

It prints a markdown table with the accuracy as mean and standard deviation over the two seeds, and the wall time of each condition:

```text
| condition | accuracy | wall time (s) |
| --- | --- | --- |
| readout | 1.0000 ± 0.0000 | 0.3 |
| frozen | 0.5388 ± 0.0243 | 7.3 |
| trained | 0.9903 ± 0.0097 | 4.0 |
```

The accuracies and the wall times can differ across machines and PyTorch versions. The frozen condition scores lowest in this small setting, where only an encoder and a readout train for 15 epochs; the numbers compare the three conditions on this task and nothing more.

## The dataset

The synthetic frequency task gives train, validation and test batches. A [`BenchmarkDataset`][torchdyno.benchmark.BenchmarkDataset] bundles them with a name and a task, here [`SequenceClassification`][torchdyno.objectives.SequenceClassification], which supplies the loss and the accuracy metric.

```python
--8<-- "examples/conditions.py:21:23"
```

## The model factory

A benchmark fits fresh models, so the model comes from a function. It is the AdaDiag composition: a gated [`EulerCell`][torchdyno.sequence.EulerCell] over a [`LeakyField`][torchdyno.sequence.LeakyField] of [`Diagonal`][torchdyno.matrices.Diagonal], [`SkewCoupling`][torchdyno.matrices.SkewCoupling] and [`Dense`][torchdyno.matrices.Dense] matrices, pooled by [`Last`][torchdyno.readout.Last] and ending in an `nn.Linear` readout. [`run_conditions()`][torchdyno.benchmark.run_conditions] requires the last module to be that readout: it freezes everything before it in the `readout` and `frozen` conditions.

```python
--8<-- "examples/conditions.py:26:41"
```

## Running the conditions

[`BenchmarkSpec`][torchdyno.benchmark.BenchmarkSpec] pairs the dataset with the seeds, 0 and 1. For each seed and condition [`run_conditions()`][torchdyno.benchmark.run_conditions] builds a model, fits it and scores it. The `readout` condition fits only the readout by ridge, choosing among three `l2` values on the validation batches. The `frozen` condition puts a linear encoder in front of the frozen parts and trains the readout by backprop with `lr` and `epochs`. The `trained` condition trains everything the same way. [`conditions_table()`][torchdyno.benchmark.conditions_table] renders the results.

```python
--8<-- "examples/conditions.py:44:51"
```

## What to try

- Add seeds, `seeds=[0, 1, 2, 3, 4]`, to tighten the standard deviations.
- Pass `ridge={"standardize": True}` to [`run_conditions()`][torchdyno.benchmark.run_conditions] to standardise the features in the readout condition.
- Add a condition of your own, `results["plasticity"] = run(spec, build)`, with [`run()`][torchdyno.benchmark.run], before printing the table.
- Return a different model from `make_model`, such as the reservoir of the [echo state network](esn.md) example.

??? example "Full code"

    ```python
    --8<-- "examples/conditions.py"
    ```

## Related concepts

- [Benchmark across seeds](../getting-started/benchmark-across-seeds.md)
- [Benchmarks](../advanced/benchmarks.md)
- [Objectives](../concepts/objectives.md)
