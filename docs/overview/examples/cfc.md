# CfC

A closed-form continuous-time cell with a small backbone, trained end to end. The model is a [`ClosedFormCell`][torchdyno.sequence.ClosedFormCell] run by [`Recurrent`][torchdyno.sequence.Recurrent], pooled by [`Last`][torchdyno.readout.Last] and read out by an `nn.Linear`, trained with backprop to classify noisy sine waves by their frequency.

Use it when you want to understand:

- [`ClosedFormCell`][torchdyno.sequence.ClosedFormCell], given any backbone module and its heads,
- how a cell can be built from ordinary `torch.nn` modules rather than from matrices,
- how the width of the heads follows from the state size.

Run it from the repository root with:

```bash
uv run python -m examples.cfc
```

It prints the accuracy on the test batches:

```text
CfC test accuracy: 1.0000
```

The last digits can differ across machines and PyTorch versions.

## The data

`frequency_task` from `examples/data.py` builds noisy sine waves of three frequencies, labelled by frequency, as time-first batches `(60, B, 1)`.

```python
--8<-- "examples/data.py:8:28"
```

## The model

The cell has a state of `H = 32` units. Its backbone is any module on the concatenation of the state and the input, `H + 1` features; here a linear layer to 128 units followed by `tanh`. The heads map the backbone to four vectors of the state size, `4 * H` features, which the cell combines in closed form over an elapsed time `dt` of 1.0 per step. Because the backbone and the heads are plain `torch.nn` modules, `torch.nn.Linear` stands in for a matrix, as it does wherever a matrix is expected. The sequence layer is [`Recurrent`][torchdyno.sequence.Recurrent], level 2, and [`Last`][torchdyno.readout.Last], level 3, keeps the final state.

```python
--8<-- "examples/cfc.py:15:28"
```

## Training and scoring

[`Backprop`][torchdyno.learners.Backprop] trains with Adam at a learning rate of `3e-3`, for 100 epochs, on the cross-entropy loss. [`evaluate()`][torchdyno.learners.evaluate] scores the test batches with [`Accuracy`][torchdyno.objectives.Accuracy].

```python
--8<-- "examples/cfc.py:29:30"
```

## What to try

- Deepen the backbone with a second `nn.Linear` and `nn.Tanh`, keeping its last width equal to the input of the heads.
- Make `dt` a per-unit tensor, `dt=torch.linspace(0.1, 2.0, H)`, so that units have different time scales.
- Reduce `units` to 32 and see how much accuracy depends on the backbone width.

??? example "Full code"

    ```python
    --8<-- "examples/cfc.py"
    ```

## Related concepts

- [Sequence maps](../concepts/sequence.md)
- [Readout](../concepts/readout.md)
- [Writing your own parts](../advanced/custom-parts.md)
- [Train a model end to end](../getting-started/train-end-to-end.md)
