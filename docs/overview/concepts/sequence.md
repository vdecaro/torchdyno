# Sequence maps

The second level maps a time-first `(T, B, F)` sequence to a `(T, B, H)`
one: a cell run over time by a layer, random convolutions, or a combination of
them. Time is always the first dimension, so a batch of one sequence is
`x.unsqueeze(1)`, and a layer written for one sequence works on a batch
without change.

Two rules hold for every layer here.

- **NaN marks a step a sequence does not have.** A timestep whose features are
    all NaN is padding, as in a batch of sequences of different lengths. Recurrent
    layers hold their state over it and output NaN there; the pooling and the
    learners skip it in turn. See [Sequences of different lengths](../advanced/variable-length.md).
- **The state is the output.** A cell returns its new state, and the layer
    that runs it returns the state after every step. There is no separate output
    to compute, and the readout reads states directly.

## Cells

A [`Cell`][torchdyno.sequence.Cell] is one step of a recurrence,
`cell(u_t, h) -> h'` on `(B, state_size)` states. Cells are built from
parts of the first level, so the update rule and the matrix structure vary
independently.

| Cell | Update | Composes |
|---|---|---|
| [`LeakyCell`][torchdyno.sequence.LeakyCell] | a leaky integrator `h' = (1 - a) h + a φ(g ⊙ (B u + W h) + c)` | echo state networks |
| [`EulerCell`][torchdyno.sequence.EulerCell] | an Euler step `h' = h + ε ⊙ f(u, h)` of a vector field, e.g. [`LeakyField`][torchdyno.sequence.LeakyField] | AdaDiag, SCN |
| [`OscillatorCell`][torchdyno.sequence.OscillatorCell] | coupled damped oscillators, positions and velocities side by side | coRNN |
| [`ClosedFormCell`][torchdyno.sequence.ClosedFormCell] | a closed-form continuous-time update from a backbone | CfC |
| [`LinearDiagonalCell`][torchdyno.sequence.LinearDiagonalCell] | `h' = a ⊙ h + s ⊙ (B u)`, with `(a, s)` from [`RingDiagonal`][torchdyno.sequence.RingDiagonal], [`ZOHDiagonal`][torchdyno.sequence.ZOHDiagonal] or [`LeakyDiagonal`][torchdyno.sequence.LeakyDiagonal], or computed from the input | LRU, S4D, ParalESN |
| [`FeedbackCell`][torchdyno.sequence.FeedbackCell] | any cell that also reads a readout of its own state | closed loops, FORCE |

Each cell checks that its matrices match the state at construction, and raises
`ValueError` with the offending width, rather than failing in the middle of
a run.

### Leaky cells

[`LeakyCell`][torchdyno.sequence.LeakyCell] takes an `input` and a `recurrent`
matrix, a `leak` rate in `(0, 1]`, constant or per unit, and an
`activation` (`torch.tanh` by default). Its gain `g` and bias `c` start
at one and zero, an exact no-op, and change only under intrinsic plasticity or
backprop. Put the units' bias in the input matrix with `Dense(..., bias=...)`
if they need one that is fitted by backprop.

```python
import torch
from torch import nn

from torchdyno.matrices import (
    BlockDiagonal,
    Dense,
    Diagonal,
    SkewCoupling,
    spectral,
    uniform,
)
from torchdyno.sequence import LeakyCell, Recurrent

def esn_cell(features, size=64, leak=0.5):
    return LeakyCell(
        input=Dense(features, size, init=uniform(1.0)),
        recurrent=Dense(size, size, init=spectral(uniform(1.0), 0.9)),
        leak=leak,
    )

x = torch.randn(100, 4, 2)
layer = Recurrent(esn_cell(2))
states = layer(x)  # (100, 4, 64)
print(states.shape)
```

A leak below one makes the state a slow average of what the cell would
otherwise compute, which suits slow signals; one is a memoryless update of the
input and the previous state.

### Continuous-time cells

[`EulerCell`][torchdyno.sequence.EulerCell] integrates a vector field `f(u, h)`
with an explicit step. The field is any callable, often a module of your own;
[`LeakyField`][torchdyno.sequence.LeakyField] is the one provided,
`-h + W φ(h) + L h + B u`, where a block-diagonal `W` and a skew-symmetric
`L` give an assembly of coupled modules. A field without a `state_size`
needs `state_size=` on the cell.

The step of an `EulerCell`, and `dt`, `gamma` and `epsilon` of an
[`OscillatorCell`][torchdyno.sequence.OscillatorCell], or `dt` of a
[`ClosedFormCell`][torchdyno.sequence.ClosedFormCell], all take the same three things: a
constant, a per-unit tensor kept as a buffer, or a callable `step(u, h)`
computed at every timestep, such as a [`Gate`][torchdyno.sequence.Gate] (a
per-unit step in `(lo, hi)` driven by the input and the state). Constants and
tensors must be positive.

```python
from torchdyno.sequence import EulerCell, Gate, LeakyField

sizes = [8] * 8
field = LeakyField(
    input=Dense(2, 64),
    recurrent=BlockDiagonal(sizes),
    coupling=SkewCoupling(sizes, pairs="ring"),
)
adaptive = Recurrent(EulerCell(field, step=Gate(2, 64)))
print(adaptive(x).shape)  # (100, 4, 64)
```

[`OscillatorCell`][torchdyno.sequence.OscillatorCell] keeps positions and velocities side
by side, so its `state_size` is twice the width of its input matrix; its
`recurrent` matrix maps the whole state to the positions, `(2H -> H)`.
[`ClosedFormCell`][torchdyno.sequence.ClosedFormCell] takes a `backbone` module on
`[h; u]` and `heads`, a matrix whose output is four equal parts, so its
state is a quarter of that width.

```python
from torchdyno.sequence import ClosedFormCell, OscillatorCell

cornn = Recurrent(OscillatorCell(input=Dense(2, 32), recurrent=Dense(64, 32)))
cfc = Recurrent(
    ClosedFormCell(
        backbone=nn.Sequential(nn.Linear(32 + 2, 64), nn.Tanh()),
        heads=nn.Linear(64, 4 * 32),
    )
)
print(cornn.cell.state_size, cfc.cell.state_size)  # 64 32
print(cornn(x).shape, cfc(x).shape)  # (100, 4, 64) (100, 4, 32)
```

### Linear diagonal cells

[`LinearDiagonalCell`][torchdyno.sequence.LinearDiagonalCell] is the recurrence
`h' = a ⊙ h + s ⊙ (B u)`. Its `transition` supplies `(a, s)`:
[`RingDiagonal`][torchdyno.sequence.RingDiagonal] draws stable complex multipliers
(LRU), [`ZOHDiagonal`][torchdyno.sequence.ZOHDiagonal] discretises a diagonal
continuous system (S4D), [`LeakyDiagonal`][torchdyno.sequence.LeakyDiagonal] puts
any [`Diagonal`][torchdyno.matrices.Diagonal] behind a leak, `a = (1 - τ) + τ λ`
and `s = τ`, for a reservoir with a linear diagonal recurrence (ParalESN), and a
module of your own can compute them from the input. When anything is complex the state is stored as real `[Re h; Im h]`,
so `state_size` is twice the number of modes. Complex storage keeps ordinary
real-valued readouts and losses working on the states.

```python
from torchdyno.matrices import polar
from torchdyno.sequence import LeakyDiagonal, LinearDiagonalCell, RingDiagonal, Scan

lru = LinearDiagonalCell(
    input=Dense(2, 32, dtype=torch.cfloat),
    transition=RingDiagonal(32, r_min=0.4, r_max=0.99),
)
print(lru.state_size)  # 64: 32 complex modes, stored as real and imaginary

eigenvalues = Diagonal(32, init=polar((0.5, 0.9)), dtype=torch.cfloat)
reservoir = LinearDiagonalCell(
    input=Dense(2, 32, init=uniform(1.0), dtype=torch.cfloat),
    transition=LeakyDiagonal(eigenvalues, leak=0.5),
).requires_grad_(False)
```

### Closed loops

[`FeedbackCell`][torchdyno.sequence.FeedbackCell] wraps a cell and feeds a readout of
its own state back as extra input: `h' = cell([u; z], h)` with
`z = readout(h)`. The inner cell's input matrix therefore takes `F + O`
features. Sharing the readout with the model's output feeds the model's own
prediction back; a `(T, B, 0)` input runs the loop with no input at all.

```python
from torchdyno.sequence import FeedbackCell

readout = nn.Linear(64, 1)
loop = nn.Sequential(
    Recurrent(
        FeedbackCell(
            LeakyCell(
                input=Dense(0 + 1, 64, init=uniform(1.0)),
                recurrent=Dense(64, 64, init=spectral(uniform(1.0), 0.9)),
            ),
            readout,
        )
    ),
    readout,
)
print(loop(torch.zeros(30, 2, 0)).shape)  # (30, 2, 1)
```

[`FORCE`][torchdyno.learners.FORCE] fits that readout while the loop runs; see
[FORCE](../examples/force.md).

## Layers

[`Recurrent`][torchdyno.sequence.Recurrent] runs a cell step by step.
[`Scan`][torchdyno.sequence.Scan] runs a
[`LinearDiagonalCell`][torchdyno.sequence.LinearDiagonalCell] by a parallel scan over time,
in `ceil(log2 T)` steps, and raises `TypeError` for a cell with no
`coefficients`. Both start from zeros, or from `h0`, and return the state
after every step, with the same results up to rounding. Use `Scan` for long
sequences of a linear cell, and `Recurrent` for everything else.

```python
states = layer(x)
torch.testing.assert_close(layer(x[50:], h0=states[49]), states[50:])

scanned, stepped = Scan(lru), Recurrent(lru)
torch.testing.assert_close(scanned(x), stepped(x), rtol=1e-4, atol=1e-5)
```

With `carry=True` a layer keeps its last state, detached, and the next call
starts from it, for truncated backprop through time or a stream fed in chunks;
`reset_state()` starts over. An explicit `h0` takes precedence over the
carried state.

```python
stream = Recurrent(esn_cell(2), carry=True)
first, second = stream(x[:50]), stream(x[50:])
torch.testing.assert_close(torch.cat([first, second]), Recurrent(stream.cell)(x))
stream.reset_state()
```

!!! warning

    A carrying layer holds state between calls, which a validation pass will
    continue from the end of training unless it is reset. Call `reset_state()`
    between sequences that should be independent.

### Compile the cell, not the model

To speed the loop up, compile the cell in place with `layer.cell.compile()`.
Compiling the whole model unrolls the loop over every timestep, which takes far
longer to compile.

## Combining layers

| Part | Does |
|---|---|
| [`Stack`][torchdyno.sequence.Stack] | runs layers in series and concatenates every layer's output; with `feed_input=True` every layer after the first also reads the input |
| [`Concat`][torchdyno.sequence.Concat] | runs modules side by side on the same input and concatenates them |
| [`Sum`][torchdyno.sequence.Sum] | runs modules side by side and adds them, e.g. a residual connection |
| [`Reverse`][torchdyno.sequence.Reverse] | runs a layer from the last step to the first, with the output aligned to the input |
| [`Diff`][torchdyno.sequence.Diff] | the first difference over time |

`Stack` is a `Recurrent` per layer, each reading the previous layer's
output, so the next layer's input width is the previous layer's state size;
the concatenation of all layers is what the readout sees. `Concat` and `Sum`
take any modules, pooling included, and `Sum` needs outputs of one shape.

```python
from torchdyno.sequence import Concat, Diff, Reverse, Stack, Sum

bidirectional = Concat(Recurrent(esn_cell(2)), Reverse(Recurrent(esn_cell(2))))
deep = Stack(Recurrent(esn_cell(2)), Recurrent(esn_cell(64)), Recurrent(esn_cell(64)))
residual = Sum(nn.Linear(2, 64), Recurrent(esn_cell(2)))
print([part(x).shape[-1] for part in (bidirectional, deep, residual)])  # [128, 192, 64]

change = Diff()(states)  # NaN at the first step, which has no predecessor
assert change[0].isnan().all()
```

!!! note

    `Diff` makes its first step NaN, so it is absent for what follows. That is
    right after a recurrent layer, but concatenating a differenced input with the
    input itself would leave rows partly NaN, which the learners reject. Difference
    after a layer, not beside the raw input.

A deep reservoir is the `Stack` of [Deep echo state network](../examples/deep_esn.md), and a
bidirectional layer reads a sequence both ways. `Reverse` shifts padding at
the end of a shorter sequence to the front, where the layer holds its initial
state over it.

## Random convolutions

[`RandomConv`][torchdyno.sequence.RandomConv] holds fixed random kernels as buffers
and maps `(T, B, C)` to `(T, B, kernels)`. It takes its kernels from a
`draw` function, the way a matrix takes an initialiser:
[`rocket()`][torchdyno.sequence.rocket] by default, or
[`minirocket()`][torchdyno.sequence.minirocket], which draws 84 fixed patterns, needs a
multiple of 84 kernels and starts from zero biases that
[`QuantileBias`][torchdyno.learners.QuantileBias] then fits. Kernels are drawn for a
series `length`, and the layer rejects shorter inputs.

A kernel without padding produces fewer outputs than the input has steps; it is
aligned to the end and the positions before it are NaN, which the pooling
skips. The convolutions are fixed buffers, so there is nothing to freeze, and
nothing for backprop to train.

```python
from torchdyno.sequence import RandomConv, minirocket

rocket = RandomConv(channels=2, length=100, kernels=1_000)
mini = RandomConv(channels=2, length=100, kernels=84 * 12, draw=minirocket)
print(rocket(x).shape, mini(x).shape)  # (100, 4, 1000) (100, 4, 1008)
```

Pool the outputs with the poolings of [Readout](readout.md), and see
[ROCKET](../examples/rocket.md) and [MiniRocket](../examples/minirocket.md) for the whole
pipelines.

## Freezing and what trains

A model is a plain `nn.Sequential`, so `requires_grad_(False)` on a layer
freezes it, whatever its cells hold. Freezing never changes initialisation: a
frozen layer keeps its random draw, and the gains and biases of a
`LeakyCell` stay available to
[`IntrinsicPlasticity`][torchdyno.learners.IntrinsicPlasticity], which updates them directly.
Which parts are frozen therefore decides what [`Backprop`][torchdyno.learners.Backprop]
trains, and the closed-form learners fit only their readout.

```python
model = nn.Sequential(Recurrent(esn_cell(2)).requires_grad_(False), nn.Linear(64, 1))
print([n for n, p in model.named_parameters() if p.requires_grad])
```

## Writing a cell

A new cell subclasses [`Cell`][torchdyno.sequence.Cell], sets `state_size`
and maps `(u, h)` to the new state; `Recurrent` then runs it, and
[`assert_layer_conforms()`][torchdyno.testing.assert_layer_conforms] checks it. See
[Writing your own parts](../advanced/custom-parts.md).
