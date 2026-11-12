# Online readout

Recursive least squares fits a reservoir's readout online, then the reservoir forecasts in closed loop. The reservoir is a frozen [`LeakyCell`][torchdyno.sequence.LeakyCell] run by [`Recurrent`][torchdyno.sequence.Recurrent]; during training it reads the true previous Lorenz state (teacher forcing), and during generation it reads its own previous prediction.

Use it when you want to understand:

- how [`RLS`][torchdyno.learners.RLS] fits a linear readout one sample at a time,
- how to feed a prediction back as the next input by hand,
- how to continue a reservoir from a given state with `h0`.

Run it from the repository root with:

```bash
uv run python -m examples.rls_forcing
```

It prints the distance between the generated and the true state after 10, 50 and 100 closed-loop steps:

```text
closed-loop error after 10, 50, 100 steps: 0.0009, 0.0108, 0.0199
```

The last digits can differ across machines and PyTorch versions.

## The data

[`lorenz_system()`][torchdyno.benchmark.lorenz_system] returns a raw trajectory, here of 3000 steps; `unsqueeze(1)` adds a batch dimension of one, giving `(T, 1, 3)`. The teacher signal is the series up to the last step and the target is the same series shifted by one, so at every step the model predicts the next state. The first 2000 steps train.

```python
--8<-- "examples/rls_forcing.py:18:21"
```

## The model

The reservoir is built as in the [echo state network](esn.md) example, with the matrices [`Dense`][torchdyno.matrices.Dense] and [`spectral()`][torchdyno.matrices.spectral] inside the cell. It is kept in its own variable, `reservoir`, because the generation loop below calls it directly, apart from the readout.

```python
--8<-- "examples/rls_forcing.py:23:30"
```

## Fitting

[`RLS`][torchdyno.learners.RLS] takes any iterable of `(x, y)` batches, so a list with one batch does. It updates the last module of the model, here the `nn.Linear`, once per row of features, starting from a ridge penalty of `l2`. `washout=100` skips the first steps, while the reservoir forgets its initial state.

```python
--8<-- "examples/rls_forcing.py:31:45"
```

## Closed loop

First the reservoir runs over the training teacher signal, and its last state is kept. Then each step feeds the previous prediction, `prediction[None]` to give it a time dimension, and passes `h0=state` so the reservoir continues instead of restarting from zeros. The readout `model[-1]` maps the new state to the next prediction. After 100 steps the error against the true series is the norm of the difference at each step.

```python
--8<-- "examples/rls_forcing.py:33:45"
```

## What to try

- Lower `forgetting` of [`RLS`][torchdyno.learners.RLS] below 1, for example 0.999, to discount older samples.
- Compare with the closed-form solution: replace `RLS(l2=1e-3, washout=100)` with `Ridge(l2=1e-3, washout=100)`, from [`Ridge`][torchdyno.learners.Ridge].
- Generate for 500 steps and see at which step the error saturates, as the Lorenz system is chaotic.
- Close the loop during training too, as the [FORCE](force.md) example does with [`FeedbackCell`][torchdyno.sequence.FeedbackCell].

??? example "Full code"

    ```python
    --8<-- "examples/rls_forcing.py"
    ```

## Related concepts

- [Read out a reservoir](../getting-started/first-reservoir.md)
- [Learners](../concepts/learners.md)
- [Sequence maps](../concepts/sequence.md)
