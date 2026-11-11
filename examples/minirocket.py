"""MiniRocket: fixed kernel patterns, biases from data, PPV and a ridge classifier.

The biases are set from quantiles of the kernels' outputs on the training
series, then the ridge penalty is chosen on a validation split. Run with
``python -m examples.minirocket``.
"""

import torch
from torch import nn

from examples.data import frequency_task
from torchdyno import seed_all
from torchdyno.learners import Chain, QuantileBias, Ridge, evaluate
from torchdyno.objectives import Accuracy
from torchdyno.readout import PPV
from torchdyno.sequence import RandomConv, minirocket

seed_all(0)
train, val, test = frequency_task()
model = nn.Sequential(
    RandomConv(channels=1, length=60, kernels=9_996, draw=minirocket),
    PPV(),
    nn.Linear(9_996, 3),
)
l2 = torch.logspace(-3, 3, 10).tolist()
Chain(QuantileBias(), Ridge(l2, standardize=True, score_fn=Accuracy())).fit(
    model, train, val
)
print(f"MiniRocket test accuracy: {evaluate(model, test, Accuracy()):.4f}")
