# Read out a reservoir

An echo state network is a fixed random recurrent network, the reservoir,
whose states a linear readout maps to outputs. Only the readout is fitted, in
closed form. Here one forecasts the Lorenz system a step ahead.

## The data

Every loader in TorchDyno is an iterable of `(x, y)` batches with time first:
`x` has shape `(T, B, F)`, `T` timesteps of `B` sequences with `F`
features. [`lorenz_benchmark()`][torchdyno.benchmark.lorenz_benchmark] cuts one trajectory into
train, validation and evaluation segments, each a single `(T, 1, 3)` batch.

```python
import torch
from torch import nn

from torchdyno import seed_all
from torchdyno.benchmark import lorenz_benchmark

seed_all(0)
data = lorenz_benchmark(length=2000, target_delay=1, washout=100)
(x, y), = data.train_loader
print(x.shape, y.shape)  # torch.Size([1000, 1, 3]) twice
```

## The model

The reservoir is a [`LeakyCell`][torchdyno.sequence.LeakyCell], a leaky-integrator
update, given its two matrices. [`Dense`][torchdyno.matrices.Dense] takes an
initialiser, and [`spectral()`][torchdyno.matrices.spectral] rescales the recurrent
matrix to a spectral radius of 0.9. [`Recurrent`][torchdyno.sequence.Recurrent] runs
the cell over time and returns the state at every step, `(T, B, 300)`, which
`nn.Linear` maps to the three outputs at every step. Freezing the reservoir
with `requires_grad_(False)` leaves its initialisation as it is.

```python
from torchdyno.matrices import Dense, spectral, uniform
from torchdyno.sequence import LeakyCell, Recurrent

reservoir = Recurrent(
    LeakyCell(
        input=Dense(3, 300, init=uniform(1.0)),
        recurrent=Dense(300, 300, init=spectral(uniform(1.0), 0.9)),
        leak=0.5,
    )
).requires_grad_(False)
esn = nn.Sequential(reservoir, nn.Linear(300, 3))
```

## Fit and score

[`Ridge`][torchdyno.learners.Ridge] fits the model's last module on whatever
reaches it. Given several penalties, it keeps the one that scores best on the
validation loader. The washout leaves out the first steps, while the reservoir
forgets its initial state.

```python
from torchdyno.learners import Ridge, evaluate

result = Ridge(
    l2=[1e-8, 1e-6, 1e-4],
    washout=data.washout,
    score_fn=data.task.primary,
).fit(esn, data.train_loader, data.val_loader)
print(result.best)  # the chosen l2 and its validation NRMSE

nrmse = evaluate(esn, data.eval_loader, data.task.primary, washout=data.washout)
print(f"test NRMSE: {nrmse:.2e}")
```

## Where to go next

Every part is replaceable on its own: try a ring reservoir with
`init=cycle(0.9)`, three reservoirs in a [`Stack`][torchdyno.sequence.Stack],
or [`IntrinsicPlasticity`][torchdyno.learners.IntrinsicPlasticity] chained before the ridge
readout. The [concepts](../concepts/index.md) list what each level offers.
