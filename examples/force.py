"""A reservoir that generates a two-tone wave on its own, trained by FORCE.

The readout's output is fed back into the reservoir, which has no other input.
FORCE fits the readout by recursive least squares at every step while the loop
runs, then the loop keeps generating with the readout fixed. Run with
``python -m examples.force``.
"""

import math

import torch
from torch import nn

from torchdyno import seed_all
from torchdyno.learners import FORCE
from torchdyno.matrices import Dense, normal, spectral, uniform
from torchdyno.sequence import FeedbackCell, LeakyCell, Recurrent

seed_all(0)
t = torch.arange(4600.0)
wave = torch.sin(2 * math.pi * t / 60) + 0.5 * torch.sin(2 * math.pi * t / 120)
target, nothing = wave[:, None, None], torch.empty(len(t), 1, 0)
train = 4000

readout = nn.Linear(300, 1)
nn.init.zeros_(readout.weight)
nn.init.zeros_(readout.bias)
loop = Recurrent(
    FeedbackCell(
        LeakyCell(
            input=Dense(1, 300, init=uniform(1.0)),
            recurrent=Dense(300, 300, init=spectral(normal(1.0), 1.2)),
            leak=0.2,
        ),
        readout,
    ),
    carry=True,
)
model = nn.Sequential(loop, readout).requires_grad_(False)
FORCE(l2=1.0).fit(model, [(nothing[:train], target[:train])])

with torch.no_grad():
    generated = model(nothing[train:])
error = (generated - target[train:]).abs().squeeze()
print(
    f"free-running error over steps 1-100, 501-600: {error[:100].mean():.4f}, "
    f"{error[500:].mean():.4f} (amplitude {wave.abs().mean():.2f})"
)
