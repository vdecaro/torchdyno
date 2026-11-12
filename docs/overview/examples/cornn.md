# coRNN

Coupled damped oscillators, trained end to end. The model is an [`OscillatorCell`][torchdyno.sequence.OscillatorCell] run by [`Recurrent`][torchdyno.sequence.Recurrent], pooled by [`Last`][torchdyno.readout.Last] and read out by an `nn.Linear`, trained with backprop to classify noisy sine waves by their frequency.

Use it when you want to understand:

- [`OscillatorCell`][torchdyno.sequence.OscillatorCell], whose state holds positions and velocities side by side,
- why the recurrent matrix reads `2 * H` features while the state has `H` units per half,
- how the damping and the stiffness are set as cell parameters.

Run it from the repository root with:

```bash
uv run python -m examples.cornn
```

It prints the accuracy on the test batches:

```text
coRNN test accuracy: 1.0000
```

The last digits can differ across machines and PyTorch versions.

## The data

`frequency_task` from `examples/data.py` builds noisy sine waves of three frequencies, labelled by frequency, as time-first batches `(60, B, 1)`.

```python
--8<-- "examples/data.py:8:28"
```

## The model

The cell integrates a second-order system of coupled damped oscillators with a step `dt` of 0.05. Its state is the positions and the velocities of `H = 64` oscillators, concatenated, so it has `2 * H` features. The recurrent matrix therefore maps `2 * H` features to `H` positions, and the input matrix maps the single input feature to `H`. Both are [`Dense`][torchdyno.matrices.Dense] maps, level 1: the input bias starts at zero and the recurrent weights come from `uniform(H**-0.5)`. `gamma` is the restoring force and `epsilon` the damping; each can also be a per-unit tensor. [`Last`][torchdyno.readout.Last], level 3, keeps the final state, `(B, 2 * H)`, and the readout maps it to the three classes.

```python
--8<-- "examples/cornn.py:16:31"
```

## Training and scoring

[`Backprop`][torchdyno.learners.Backprop] trains every parameter with Adam, at a learning rate of `1e-2`, for 30 epochs on the cross-entropy loss. [`evaluate()`][torchdyno.learners.evaluate] scores the test batches with [`Accuracy`][torchdyno.objectives.Accuracy].

```python
--8<-- "examples/cornn.py:32:33"
```

## What to try

- Change `dt` to 0.1 or 0.01 to see how the integration step trades speed for stability.
- Change `gamma` and `epsilon`, and see how the restoring force and the damping affect the accuracy.
- Pool with [`Mean`][torchdyno.readout.Mean] instead of [`Last`][torchdyno.readout.Last].
- Raise `H` to 128, and change the readout to `nn.Linear(2 * H, 3)` accordingly.

??? example "Full code"

    ```python
    --8<-- "examples/cornn.py"
    ```

## Related concepts

- [Sequence maps](../concepts/sequence.md)
- [Readout](../concepts/readout.md)
- [Train a model end to end](../getting-started/train-end-to-end.md)
