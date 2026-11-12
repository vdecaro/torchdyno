# ParalESN

Three reservoir layers of linear complex modes, from Pinna et al., "ParalESN: Enabling parallel information processing in Reservoir Computing" (2026), forecast the Lorenz system with a single ridge readout over every layer. Each layer's recurrence is diagonal and linear, so [`Scan`][torchdyno.sequence.Scan] runs it over the whole sequence in parallel; a convolution across the modes and a tanh after it are the only nonlinearity. Only the readout is fitted.

Use it when you want to understand:

- how [`LeakyDiagonal`][torchdyno.sequence.LeakyDiagonal] turns a [`Diagonal`][torchdyno.matrices.Diagonal] drawn by [`polar()`][torchdyno.matrices.polar] into a leaky reservoir,
- how [`Mix`][torchdyno.sequence.Mix] couples the modes that a diagonal recurrence leaves apart,
- how a layer of the sequence level can be a small `nn.Sequential` of its own.

Run it from the repository root with:

```bash
uv run python -m examples.paralesn
```

It prints the NRMSE on the evaluation segment:

```text
ParalESN test NRMSE: 3.10e-04
```

The last digits can differ across machines and PyTorch versions.

## The layer helper

A function builds one layer. The eigenvalues are a complex [`Diagonal`][torchdyno.matrices.Diagonal], drawn by [`polar()`][torchdyno.matrices.polar] with a modulus between 0.5 and 0.9 and a phase anywhere on the circle, so every mode is stable and oscillates at its own frequency. Each mode reads its input scaled by `sqrt(1 - |λ|²)`, as in the paper, so slow modes, which accumulate more, read less; the input matrix also carries a small complex bias. [`LeakyDiagonal`][torchdyno.sequence.LeakyDiagonal] puts the eigenvalues behind a leak of 0.5, and [`LinearDiagonalCell`][torchdyno.sequence.LinearDiagonalCell], run by [`Scan`][torchdyno.sequence.Scan], returns the real and imaginary parts of every mode side by side.

[`Mix`][torchdyno.sequence.Mix] with `channels=2` reads those two blocks and convolves across neighbouring modes, the real part of a complex convolution, and `nn.Tanh()` follows. The layer's output is one real value per mode, which the next layer reads.

```python
--8<-- "examples/paralesn.py:19:38"
```

## The model

[`Stack`][torchdyno.sequence.Stack] runs the three layers in series and returns their outputs side by side, `(T, B, 600)`, the width of the readout, as in the [deep echo state network](deep_esn.md). The stack is frozen as a whole: the eigenvalues, the input matrices and the mixing kernels keep their random draws.

```python
--8<-- "examples/paralesn.py:41:46"
```

## Fitting and scoring

[`Ridge`][torchdyno.learners.Ridge] fits the readout on the training loader after the washout, and [`evaluate()`][torchdyno.learners.evaluate] scores the evaluation loader with the dataset's primary metric. The task and the sizes are those of the [deep echo state network](deep_esn.md), whose layers have dense recurrent matrices and step through time one by one.

```python
--8<-- "examples/paralesn.py:47:50"
```

## What to try

- Narrow the phases, `polar((0.5, 0.9), (0.0, 0.5))`, to keep only slowly rotating modes, and compare the error.
- Widen the kernel of [`Mix`][torchdyno.sequence.Mix], or set `kernel_size=1` so that each mode stays on its own.
- Replace [`Scan`][torchdyno.sequence.Scan] by [`Recurrent`][torchdyno.sequence.Recurrent] to step the same cells through time, with the same states.
- Read only the last time step for classification, with [`Last`][torchdyno.readout.Last] before the readout.

??? example "Full code"

    ```python
    --8<-- "examples/paralesn.py"
    ```

## Related concepts

- [Matrices](../concepts/matrices.md)
- [Sequence maps](../concepts/sequence.md)
- [Learners](../concepts/learners.md)
