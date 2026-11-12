# MiniRocket

MiniRocket's fixed kernel patterns, their biases set from quantiles of the kernels' outputs on the training series, PPV pooling and a ridge classifier. The kernels are a [`RandomConv`][torchdyno.sequence.RandomConv] drawn by [`minirocket()`][torchdyno.sequence.minirocket], and two learners are chained: [`QuantileBias`][torchdyno.learners.QuantileBias], then [`Ridge`][torchdyno.learners.Ridge].

Use it when you want to understand:

- how [`minirocket()`][torchdyno.sequence.minirocket] plugs into [`RandomConv`][torchdyno.sequence.RandomConv] as its draw,
- how [`QuantileBias`][torchdyno.learners.QuantileBias] sets biases from data,
- how [`Chain`][torchdyno.learners.Chain] runs two learners on one model.

Run it from the repository root with:

```bash
uv run python -m examples.minirocket
```

It prints the accuracy on the test batches:

```text
MiniRocket test accuracy: 1.0000
```

The last digits can differ across machines and PyTorch versions.

## The data

`frequency_task` from `examples/data.py` builds noisy sine waves of three frequencies, labelled by frequency, as time-first batches `(60, B, 1)`.

```python
--8<-- "examples/data.py:8:28"
```

## The model

`draw=minirocket` replaces the default ROCKET draw, the way a matrix takes an initialiser. MiniRocket has 84 fixed patterns at spread dilations, so `kernels` must be a multiple of 84; 9 996 is 84 times 119. The biases start at zero. A single [`PPV`][torchdyno.readout.PPV], level 3, pools each kernel to one feature, so the `nn.Linear` readout has 9 996 inputs.

```python
--8<-- "examples/minirocket.py:18:24"
```

## Fitting

[`Chain`][torchdyno.learners.Chain] runs its learners one after the other on the same model. [`QuantileBias`][torchdyno.learners.QuantileBias] first sets each kernel's bias to minus a quantile of its outputs on a training series, so that its PPV is about one minus that quantile. [`Ridge`][torchdyno.learners.Ridge] then standardises the features, picks its penalty among ten values on the validation batches, and fits the readout.

```python
--8<-- "examples/minirocket.py:25:28"
```

## Scoring

```python
--8<-- "examples/minirocket.py:29:29"
```

## What to try

- Drop [`QuantileBias`][torchdyno.learners.QuantileBias] from the chain, leaving the biases at zero, to see what the data-driven biases add.
- Use fewer kernels, 84 times a whole number such as 2 520, and set the readout width to match.
- Compare with the [ROCKET](rocket.md) example, which draws random biases and adds max pooling.

??? example "Full code"

    ```python
    --8<-- "examples/minirocket.py"
    ```

## Related concepts

- [Sequence maps](../concepts/sequence.md)
- [Readout](../concepts/readout.md)
- [Learners](../concepts/learners.md)
