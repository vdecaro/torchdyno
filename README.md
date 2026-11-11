<p align="center">
<img width=200 alt="torchdyno-logo" src="https://raw.githubusercontent.com/vdecaro/torchdyno/refs/heads/main/docs/_static/images/logo.png"/>
</p>

![PyPI - Version](https://img.shields.io/pypi/v/torchdyno)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchdyno)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&)](https://github.com/pre-commit/pre-commit)
[![codecov](https://codecov.io/github/vdecaro/torchdyno/graph/badge.svg)](https://codecov.io/github/vdecaro/torchdyno)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/vdecaro/torchdyno/main.svg)](https://results.pre-commit.ci/latest/github/vdecaro/torchdyno/main)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Import sorted with isort](https://img.shields.io/badge/isort-checked-brightgreen)](https://pycqa.github.io/isort/)

**TorchDyno** is a PyTorch-based library for dynamical systems.

## Installation

To install TorchDyno and all its dependencies, you can run the following commands:

```bash

pip install torchdyno

```

**N.B.** We **strongly** suggest the installation of TorchDyno in a virtual environment.

## Usage

A model is a plain `torch.nn.Sequential` of parts, which come in four kinds:

- `torchdyno.matrices` defines weight matrices: dense, diagonal, block-diagonal and skew-symmetric couplings, with their initialisers.
- `torchdyno.cells` and `torchdyno.layers` define how a sequence is processed: recurrent cells run step by step or with a parallel scan, and random convolution kernels.
- `torchdyno.pooling` reduces a sequence over time; an `nn.Linear` maps the result to outputs.
- `torchdyno.learners`, `torchdyno.tasks` and `torchdyno.metrics` fit and score the model: ridge regression, recursive least squares, backprop and intrinsic plasticity.

An assembly of diagonal modules with an adaptive step, trained end to end:

```python
import torch
from torch import nn

from torchdyno.cells import EulerCell, Gate
from torchdyno.layers import Recurrent
from torchdyno.learners import Backprop
from torchdyno.matrices import Dense, Diagonal, SkewCoupling, normal
from torchdyno.pooling import Last

H = 16 * 8
model = nn.Sequential(
    Recurrent(
        EulerCell(
            recurrent=Diagonal(H, bound=torch.tanh),
            coupling=SkewCoupling([16] * 8, pairs=8),
            input=Dense(1, H, init=normal(H**-0.5)),
            step=Gate(1, H, bounds=(1e-5, 0.2)),
            activation=torch.relu,
        )
    ),
    Last(),
    nn.Linear(H, 3),
)
Backprop(nn.CrossEntropyLoss(), lr=1e-2, epochs=30).fit(model, train)
```

The same parts compose other models. ROCKET's features read out by ridge regression:

```python
from torchdyno.layers import Concat, RandomConv
from torchdyno.learners import Ridge
from torchdyno.metrics import Accuracy
from torchdyno.pooling import PPV, Max

rocket = nn.Sequential(
    RandomConv(channels=1, length=60, kernels=10_000),
    Concat(PPV(), Max()),
    nn.Linear(20_000, 3),
)
Ridge(
    torch.logspace(-3, 3, 10).tolist(),
    standardize=True,
    score_fn=Accuracy(),
).fit(rocket, train, val)
```

[`examples/`](examples) composes echo state networks, AdaDiag, SCN, coRNN, CfC, LRU, S4D and ROCKET. Run one from the repository root with `python -m examples.<name>`, e.g. `python -m examples.esn`.

## Documentation

The documentation for TorchDyno can be found [here](https://torchdyno.readthedocs.io/en/latest/).

## Maintainers

TorchDyno is maintained by [Valerio De Caro](https://github.com/vdecaro).

## Contact Us

If you want to get in touch with us, [drop us an e-mail](mailto:valerio@valeriodecaro.com?subject=[TorchDyno]%20Request%20for%20information)!
