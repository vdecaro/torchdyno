"""ROCKET: random convolution kernels, PPV and max pooling, and a ridge classifier.

As in the paper, features are standardised and the ridge penalty is chosen
from ``logspace(-3, 3, 10)``, here on a validation split. Run with
``python -m examples.rocket``.
"""

import torch
from torch import nn

from examples.data import frequency_task
from torchdyno import seed_all
from torchdyno.layers import Concat, RandomConv
from torchdyno.learners import Ridge, evaluate
from torchdyno.metrics import Accuracy
from torchdyno.pooling import PPV, Max

seed_all(0)
train, val, test = frequency_task()
rocket = nn.Sequential(
    RandomConv(channels=1, length=60, kernels=10_000),
    Concat(PPV(), Max()),
    nn.Linear(20_000, 3),
)
l2 = torch.logspace(-3, 3, 10).tolist()
Ridge(l2, standardize=True, score_fn=Accuracy(), mode="max").fit(rocket, train, val)
print(f"ROCKET test accuracy: {evaluate(rocket, test, Accuracy()):.4f}")
