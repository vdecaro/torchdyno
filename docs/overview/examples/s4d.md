# S4D

S4D as a frozen feature extractor, pooled like ROCKET and read out by ridge. The modes are a [`LinearDiagonalCell`][torchdyno.sequence.LinearDiagonalCell] with a [`ZOHDiagonal`][torchdyno.sequence.ZOHDiagonal] transition, run by [`Scan`][torchdyno.sequence.Scan] and frozen; PPV and max pooling turn the states into features, and only the readout is fitted, to classify noisy sine waves by their frequency.

Use it when you want to understand:

- [`ZOHDiagonal`][torchdyno.sequence.ZOHDiagonal], whose eigenvalues come from a zero-order hold over a range of time steps,
- how [`Concat`][torchdyno.sequence.Concat] pools a sequence two ways at once,
- how the same composition can train end to end or be fitted by ridge.

Run it from the repository root with:

```bash
uv run python -m examples.s4d
```

It prints the accuracy on the test batches:

```text
S4D test accuracy: 1.0000
```

The last digits can differ across machines and PyTorch versions.

## The data

`frequency_task` from `examples/data.py` builds noisy sine waves of three frequencies, labelled by frequency. This example uses the validation batches too, to choose the ridge penalty.

```python
--8<-- "examples/data.py:8:28"
```

## The model

The cell has `N = 128` complex modes: a complex [`Dense`][torchdyno.matrices.Dense] input and a [`ZOHDiagonal`][torchdyno.sequence.ZOHDiagonal] transition, whose time steps range from `dt_min` to `dt_max` so the modes cover different time scales. [`Scan`][torchdyno.sequence.Scan] runs it in parallel and `requires_grad_(False)` freezes it.

The modes are linear in the input, so the nonlinearity has to come from the pooling. [`Concat`][torchdyno.sequence.Concat] runs [`PPV`][torchdyno.readout.PPV], the proportion of positive values over time, and [`Max`][torchdyno.readout.Max] side by side on the states, level 3. Each pools the `2 * N` real and imaginary features, giving `4 * N` features for the `nn.Linear` readout.

```python
--8<-- "examples/s4d.py:20:32"
```

## Fitting and scoring

[`Ridge`][torchdyno.learners.Ridge] standardises the features, tries seven penalties from `1e-4` to `1e2` and keeps the one with the best accuracy on the validation batches. [`evaluate()`][torchdyno.learners.evaluate] scores the test batches.

```python
--8<-- "examples/s4d.py:33:35"
```

## What to try

- Train the same composition end to end: drop `requires_grad_(False)` and fit with [`Backprop`][torchdyno.learners.Backprop], as the [LRU](lru.md) example does.
- Pool with [`Mean`][torchdyno.readout.Mean] only, and see how the nonlinear pooling matters.
- Narrow the range `dt_min` to `dt_max` and watch the accuracy as the time scales concentrate.
- Raise `N` to 256, which makes the readout `nn.Linear(4 * N, 3)` take 1 024 features.

??? example "Full code"

    ```python
    --8<-- "examples/s4d.py"
    ```

## Related concepts

- [Sequence maps](../concepts/sequence.md)
- [Readout](../concepts/readout.md)
- [Learners](../concepts/learners.md)
