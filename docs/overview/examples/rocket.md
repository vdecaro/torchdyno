# ROCKET

Ten thousand random convolution kernels, PPV and max pooling, and a ridge classifier on standardised features. The kernels are a [`RandomConv`][torchdyno.sequence.RandomConv], frozen by construction, and the readout is fitted by ridge on a synthetic task of noisy sine waves classified by their frequency.

Use it when you want to understand:

- how [`RandomConv`][torchdyno.sequence.RandomConv] draws and holds a bank of fixed kernels,
- how [`Concat`][torchdyno.sequence.Concat] joins PPV and max pooling,
- how ridge selects its penalty on a validation split with standardised features.

Run it from the repository root with:

```bash
uv run python -m examples.rocket
```

It prints the accuracy on the test batches:

```text
ROCKET test accuracy: 1.0000
```

The last digits can differ across machines and PyTorch versions.

## The data

`frequency_task` from `examples/data.py` builds noisy sine waves of three frequencies, labelled by frequency, as time-first batches `(60, B, 1)`.

```python
--8<-- "examples/data.py:8:28"
```

## The model

[`RandomConv`][torchdyno.sequence.RandomConv] takes the number of `channels`, the series `length` and the number of `kernels`, and draws them with [`rocket()`][torchdyno.sequence.rocket], its default. Kernels are kept as buffers, so nothing needs freezing. A kernel without padding leaves NaN at the steps it does not cover, which the pooling skips. [`PPV`][torchdyno.readout.PPV] and [`Max`][torchdyno.readout.Max], level 3, are run side by side by [`Concat`][torchdyno.sequence.Concat], giving two features per kernel, 20 000 in all, for the `nn.Linear` readout.

```python
--8<-- "examples/rocket.py:18:24"
```

## Fitting and scoring

[`Ridge`][torchdyno.learners.Ridge] standardises the features, tries ten penalties from `1e-3` to `1e3`, and keeps the one with the best accuracy on the validation batches. [`evaluate()`][torchdyno.learners.evaluate] scores the test batches with [`Accuracy`][torchdyno.objectives.Accuracy].

```python
--8<-- "examples/rocket.py:25:27"
```

## What to try

- Reduce `kernels` to 1 000, and change the readout to `nn.Linear(2_000, 3)`, to see how little accuracy this task needs.
- Pool with [`PPV`][torchdyno.readout.PPV] alone, and compare.
- Swap the kernel draw for MiniRocket's: `draw=minirocket`, as in the [MiniRocket](minirocket.md) example.
- Pass `class_weight="balanced"` to [`Ridge`][torchdyno.learners.Ridge] on an unbalanced dataset.

??? example "Full code"

    ```python
    --8<-- "examples/rocket.py"
    ```

## Related concepts

- [Sequence maps](../concepts/sequence.md)
- [Readout](../concepts/readout.md)
- [Learners](../concepts/learners.md)
- [Benchmarks](../advanced/benchmarks.md)
