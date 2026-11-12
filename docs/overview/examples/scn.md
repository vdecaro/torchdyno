# SCN

Fixed block-diagonal modules coupled by a trained skew-symmetric matrix. The blocks are frozen, while the coupling, the input and the readout train. The model is an [`EulerCell`][torchdyno.sequence.EulerCell] run by [`Recurrent`][torchdyno.sequence.Recurrent], trained with backprop to classify noisy sine waves by their frequency.

Use it when you want to understand:

- how [`BlockDiagonal`][torchdyno.matrices.BlockDiagonal] and [`SkewCoupling`][torchdyno.matrices.SkewCoupling] split a state into modules and couple them,
- how to freeze one part of a model and train the rest,
- how to reach a part inside a model to freeze it.

Run it from the repository root with:

```bash
uv run python -m examples.scn
```

It prints the accuracy on the test batches:

```text
SCN test accuracy: 0.9709
```

The last digits can differ across machines and PyTorch versions.

## The data

`frequency_task` from `examples/data.py` builds noisy sine waves of three frequencies, labelled by frequency, as time-first batches `(60, B, 1)`. The example uses the train and test batches.

```python
--8<-- "examples/data.py:8:28"
```

## The model

The recurrent matrix is [`BlockDiagonal`][torchdyno.matrices.BlockDiagonal] with 8 blocks of 16 units, bounded by `torch.tanh`. The coupling is [`SkewCoupling`][torchdyno.matrices.SkewCoupling] over the same blocks, with `pairs="ring"` coupling each block to the next. Both are matrices, level 1, and both go into a [`LeakyField`][torchdyno.sequence.LeakyField], together with a [`Dense`][torchdyno.matrices.Dense] input. [`EulerCell`][torchdyno.sequence.EulerCell] integrates the field with a constant step of 0.1. [`Last`][torchdyno.readout.Last] keeps the final state and `nn.Linear` maps it to the three classes.

```python
--8<-- "examples/scn.py:18:34"
```

## Freezing the blocks

`scn[0]` is the [`Recurrent`][torchdyno.sequence.Recurrent] layer, `.cell.field` the field, and `.recurrent` the block-diagonal matrix. `requires_grad_(False)` freezes it: its values stay as initialised while the rest trains. Freezing never changes how a matrix is initialised.

```python
--8<-- "examples/scn.py:35:35"
```

## Training

[`Backprop`][torchdyno.learners.Backprop] trains whatever still requires grad, here the coupling, the input matrix and the readout, with Adam at a learning rate of `1e-2` for 30 epochs. [`evaluate()`][torchdyno.learners.evaluate] scores the test batches with [`Accuracy`][torchdyno.objectives.Accuracy].

```python
--8<-- "examples/scn.py:36:37"
```

## What to try

- Remove the freezing line to train the blocks too, and compare the accuracy.
- Replace the ring with `pairs=8` or `pairs=0.5`, which couples a number or a share of randomly chosen pairs.
- Freeze the coupling instead and train the blocks: `scn[0].cell.field.coupling.requires_grad_(False)`.

??? example "Full code"

    ```python
    --8<-- "examples/scn.py"
    ```

## Related concepts

- [Matrices](../concepts/matrices.md)
- [Sequence maps](../concepts/sequence.md)
- [Train a model end to end](../getting-started/train-end-to-end.md)
