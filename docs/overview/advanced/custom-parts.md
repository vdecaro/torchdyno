# Writing your own parts

Each level is open: a part only has to follow its level's contract, and every
other part, layer and learner works with it.

## A matrix

A matrix is any module with `in_features`, `out_features`, a
`forward(x)` that applies `x Wᵀ`, and a `dense()` that returns `W`,
which [`to_dense()`][torchdyno.matrices.to_dense], [`update_()`][torchdyno.matrices.update_]
and [`spectral_radius()`][torchdyno.matrices.spectral_radius] read.

```python
import torch
from torch import nn

from torchdyno.matrices import Dense, spectral_radius, update_

class LowRank(nn.Module):
    """W = U Vᵀ, applied without building W."""

    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.u = nn.Parameter(torch.randn(out_features, rank) / rank**0.5)
        self.v = nn.Parameter(torch.randn(in_features, rank) / in_features**0.5)

    def forward(self, x):
        return (x @ self.v) @ self.u.T

    def dense(self):
        return self.u @ self.v.T

low_rank = LowRank(64, 64, rank=4)
update_(low_rank, 1e-2 * torch.eye(64))  # pulled back to U and V
print(spectral_radius(low_rank))
```

## A cell

A cell subclasses [`Cell`][torchdyno.sequence.Cell], sets its `state_size`,
and maps `(u, h)` to the new state. [`assert_layer_conforms()`][torchdyno.testing.assert_layer_conforms]
checks a layer running it against the plain loop over the cell: the shapes and
dtype, each sequence of the batch running on its own, continuing from `h0`,
and gradients reaching the input and every trainable parameter.

```python
from torchdyno.sequence import Cell, Recurrent
from torchdyno.testing import assert_layer_conforms, echo_state_distance

class TanhCell(Cell):
    """h' = tanh(W h + B u)."""

    def __init__(self, input, recurrent):
        super().__init__()
        self.input, self.recurrent = input, recurrent
        self.state_size = recurrent.out_features

    def forward(self, u, h):
        return torch.tanh(self.recurrent(h) + self.input(u))

layer = Recurrent(TanhCell(Dense(2, 64), low_rank))
assert_layer_conforms(layer, torch.randn(10, 3, 2))
print(echo_state_distance(layer, torch.randn(200, 3, 2))[-1])
```

[`echo_state_distance()`][torchdyno.testing.echo_state_distance] runs a layer from two random
starts: with the echo state property the distance between the runs decays to
zero.

## A kernel draw

[`RandomConv`][torchdyno.sequence.RandomConv] takes its kernels from any
`draw(channels, length, kernels, dtype)` that returns
[`Kernels`][torchdyno.sequence.Kernels]:

```python
from torchdyno.sequence import Kernels, RandomConv

def short(channels, length, kernels, dtype):
    return Kernels(
        weight=torch.randn(kernels, channels, 3, dtype=dtype),
        bias=torch.zeros(kernels, dtype=dtype),
        lengths=torch.full((kernels,), 3),
        dilations=torch.ones(kernels, dtype=torch.long),
        paddings=torch.ones(kernels, dtype=torch.long),
    )

conv = RandomConv(channels=2, length=20, kernels=100, draw=short)
```

## A learner

A learner is any object with `fit(model, train, val=None) -> FitResult`, so
a local rule fits in a few lines and chains with the others:

```python
from torchdyno.learners import Chain, FitResult, Ridge

class Hebbian:
    """Strengthen each recurrent weight by the correlation of its units."""

    def __init__(self, lr=1e-3):
        self.lr = lr

    def fit(self, model, train, val=None):
        layer = model[0]
        for x, _ in train:
            with torch.no_grad():
                h = layer(x).flatten(0, 1)
            update_(layer.cell.recurrent, self.lr * h.T @ h / len(h))
        return FitResult()

model = nn.Sequential(layer, nn.Linear(64, 1))
x = torch.randn(100, 3, 2)
Chain(Hebbian(), Ridge(1e-6)).fit(model, [(x, x[..., :1])])
```
