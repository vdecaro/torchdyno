# LRU

A linear complex diagonal recurrence run with a parallel scan, trained end to end. The model is a [`LinearDiagonalCell`][torchdyno.sequence.LinearDiagonalCell] run by [`Scan`][torchdyno.sequence.Scan], followed by a position-wise nonlinearity, [`Mean`][torchdyno.readout.Mean] pooling and a readout, trained with backprop to classify noisy sine waves by their frequency.

Use it when you want to understand:

- how [`RingDiagonal`][torchdyno.sequence.RingDiagonal] parametrises complex eigenvalues in a ring,
- how [`Scan`][torchdyno.sequence.Scan] replaces the step-by-step loop by a parallel scan,
- why a linear recurrence needs a nonlinearity after it.

Run it from the repository root with:

```bash
uv run python -m examples.lru
```

It prints the accuracy on the test batches:

```text
LRU test accuracy: 1.0000
```

The last digits can differ across machines and PyTorch versions.

## The data

`frequency_task` from `examples/data.py` builds noisy sine waves of three frequencies, labelled by frequency, as time-first batches `(60, B, 1)`.

```python
--8<-- "examples/data.py:8:28"
```

## The model

The cell has `N = 64` complex modes. The input matrix is a [`Dense`][torchdyno.matrices.Dense] with `dtype=torch.cfloat`, a complex matrix, level 1. [`RingDiagonal`][torchdyno.sequence.RingDiagonal] is the transition: it draws each eigenvalue with a modulus between `r_min` and `r_max` and a phase up to `max_phase`, so every mode is stable and decays at its own rate. Because the recurrence is linear in the state, [`Scan`][torchdyno.sequence.Scan] evaluates it in about `log2 T` parallel steps instead of `T` sequential ones.

The scan returns the real and imaginary parts side by side, `2 * N` features per step. The recurrence is linear, so on its own the model could not tell frequencies apart; a position-wise `nn.Linear` and `nn.GELU` supply the nonlinearity, and [`Mean`][torchdyno.readout.Mean], level 3, averages over time before the readout.

```python
--8<-- "examples/lru.py:21:35"
```

## Training and scoring

[`Backprop`][torchdyno.learners.Backprop] trains every parameter with Adam, at `1e-2`, for 30 epochs on the cross-entropy loss. [`evaluate()`][torchdyno.learners.evaluate] scores the test batches with [`Accuracy`][torchdyno.objectives.Accuracy].

```python
--8<-- "examples/lru.py:36:37"
```

## What to try

- Swap [`RingDiagonal`][torchdyno.sequence.RingDiagonal] for [`ZOHDiagonal`][torchdyno.sequence.ZOHDiagonal], as the [S4D](s4d.md) example does.
- Narrow the ring, `r_min=0.9`, to give every mode a long memory, and compare the accuracy.
- Replace [`Scan`][torchdyno.sequence.Scan] by [`Recurrent`][torchdyno.sequence.Recurrent] over the same cell to check that both give the same states, only faster or slower.
- Pool with [`Last`][torchdyno.readout.Last] instead of [`Mean`][torchdyno.readout.Mean].

??? example "Full code"

    ```python
    --8<-- "examples/lru.py"
    ```

## Related concepts

- [Matrices](../concepts/matrices.md)
- [Sequence maps](../concepts/sequence.md)
- [Readout](../concepts/readout.md)
- [Train a model end to end](../getting-started/train-end-to-end.md)
