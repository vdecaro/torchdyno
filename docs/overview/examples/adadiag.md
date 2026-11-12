# AdaDiag

A sparse assembly of diagonal modules with an adaptive step, trained end to end. Each unit has a bounded diagonal self-weight, the modules are coupled by a skew-symmetric matrix, and a gate sets every unit's Euler step from the input and the state. The model is trained with backprop to classify noisy sine waves by their frequency.

Use it when you want to understand:

- how [`EulerCell`][torchdyno.sequence.EulerCell] and [`LeakyField`][torchdyno.sequence.LeakyField] compose a continuous-time model,
- how [`Diagonal`][torchdyno.matrices.Diagonal] and [`SkewCoupling`][torchdyno.matrices.SkewCoupling] structure the recurrence,
- how [`Gate`][torchdyno.sequence.Gate] makes the step size adaptive.

Run it from the repository root with:

```bash
uv run python -m examples.adadiag
```

It prints the accuracy on the test batches:

```text
AdaDiag test accuracy: 1.0000
```

The last digits can differ across machines and PyTorch versions.

## The data

`frequency_task` builds noisy sine waves of three frequencies, and the class of a wave is its frequency. It returns train, validation and test lists of time-first batches, `x` of shape `(60, B, 1)` and integer labels `y` of shape `(B,)`. This example does not need the validation set.

```python
--8<-- "examples/data.py:8:28"
```

## The model

The cell is built from a vector field and a step. [`LeakyField`][torchdyno.sequence.LeakyField] combines three matrices, level 1. The recurrent matrix is [`Diagonal`][torchdyno.matrices.Diagonal], a self-weight per unit, bounded by `torch.tanh` so it stays in `(-1, 1)`. [`SkewCoupling`][torchdyno.matrices.SkewCoupling] couples 8 blocks of 16 units, with 8 pairs of blocks coupled, and stores only those pairs; the dense coupling `C - Cᵀ` stays skew-symmetric during training. The input matrix is a [`Dense`][torchdyno.matrices.Dense] drawn from [`normal()`][torchdyno.matrices.normal].

[`EulerCell`][torchdyno.sequence.EulerCell] integrates the field with an explicit Euler step. Here the step is a [`Gate`][torchdyno.sequence.Gate], a per-unit step size between `1e-5` and `0.2` driven by the input and the state. [`Recurrent`][torchdyno.sequence.Recurrent] runs the cell over time, and [`Last`][torchdyno.readout.Last], level 3, keeps the final state. An `nn.Linear` maps it to the three classes.

```python
--8<-- "examples/adadiag.py:19:36"
```

## Training

[`Backprop`][torchdyno.learners.Backprop] trains every parameter that requires grad, with Adam by default, for 30 epochs, minimising the cross-entropy loss. [`evaluate()`][torchdyno.learners.evaluate] then scores the test batches with [`Accuracy`][torchdyno.objectives.Accuracy], a metric of the objectives level.

```python
--8<-- "examples/adadiag.py:37:38"
```

## What to try

- Use `pairs="ring"` in [`SkewCoupling`][torchdyno.matrices.SkewCoupling] to couple each block to its neighbour, as the [SCN](scn.md) example does.
- Replace the gate with a constant step, `step=0.1`, to see what the adaptive step adds.
- Change `bounds` of the gate, or the `bound` of the diagonal, and compare the accuracy.
- Run the three training conditions side by side in [Readout, frozen, trained](conditions.md).

??? example "Full code"

    ```python
    --8<-- "examples/adadiag.py"
    ```

## Related concepts

- [Matrices](../concepts/matrices.md)
- [Sequence maps](../concepts/sequence.md)
- [Readout](../concepts/readout.md)
- [Train a model end to end](../getting-started/train-end-to-end.md)
