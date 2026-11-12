# FORCE

A reservoir with no input learns by FORCE to generate a two-tone wave. The readout's output is fed back into the reservoir, through a [`FeedbackCell`][torchdyno.sequence.FeedbackCell] that wraps a [`LeakyCell`][torchdyno.sequence.LeakyCell]. [`FORCE`][torchdyno.learners.FORCE] fits the readout by recursive least squares at every step while the loop runs, and afterwards the loop keeps generating with the readout fixed.

Use it when you want to understand:

- how [`FeedbackCell`][torchdyno.sequence.FeedbackCell] closes the loop from a readout back to a reservoir,
- how the same readout module sits inside the cell and at the end of the model,
- how [`FORCE`][torchdyno.learners.FORCE] differs from fitting a readout on teacher-forced states.

Run it from the repository root with:

```bash
uv run python -m examples.force
```

It prints the mean absolute error of the free-running wave over the first 100 steps and over steps 501 to 600, next to the mean amplitude of the wave:

```text
free-running error over steps 1-100, 501-600: 0.0332, 0.0371 (amplitude 0.68)
```

The last digits can differ across machines and PyTorch versions.

## The target

The target is a sum of two sines with periods 60 and 120, 4600 steps long. The reservoir has no input, so the input is an empty tensor with zero features, `(T, 1, 0)`. The first 4000 steps train and the remaining 600 are generated.

```python
--8<-- "examples/force.py:19:23"
```

## The model

The readout is an `nn.Linear` starting at zero, so the loop starts with no feedback. [`FeedbackCell`][torchdyno.sequence.FeedbackCell] receives the leaky cell and that same readout, and gives the cell `[u; z]`, the input followed by the fed-back output. The cell's input matrix therefore has one feature, the fed-back output, and the recurrent matrix has a spectral radius of 1.2, drawn from [`normal()`][torchdyno.matrices.normal]. `carry=True` makes [`Recurrent`][torchdyno.sequence.Recurrent] keep its last state between calls, so the generation continues where training stopped. The final `requires_grad_(False)` keeps backprop away: the learner updates the readout itself.

```python
--8<-- "examples/force.py:25:39"
```

## Training

[`FORCE`][torchdyno.learners.FORCE] finds the feedback cell in the model and updates its readout at every timestep on the new state and the target, so the fed-back output stays close to the target from the first step. `l2` sets the initial `P = I / l2`.

```python
--8<-- "examples/force.py:40:40"
```

## Free running

With the readout fixed, the model is called on 600 empty inputs and generates the wave on its own. The mean absolute error is taken over two windows, the start and the end, to show that the loop has not drifted.

```python
--8<-- "examples/force.py:42:48"
```

## What to try

- Change the periods of the wave, or add a third tone, and see how large the reservoir must be.
- Raise the leak from 0.2, or lower the spectral radius from 1.2, to see the loop lose its ability to sustain the wave.
- Fit with [`RLS`][torchdyno.learners.RLS] on the teacher-forced states instead, as in the [online readout](rls_forcing.md) example, and compare the closed-loop error.

??? example "Full code"

    ```python
    --8<-- "examples/force.py"
    ```

## Related concepts

- [Sequence maps](../concepts/sequence.md)
- [Learners](../concepts/learners.md)
- [Read out a reservoir](../getting-started/first-reservoir.md)
