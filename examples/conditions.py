"""One AdaDiag model run three ways on the same seeds: readout, frozen and trained.

Run with ``python -m examples.conditions``.
"""

import torch
from torch import nn

from examples.data import frequency_task
from torchdyno.benchmark import (
    BenchmarkDataset,
    BenchmarkSpec,
    conditions_table,
    run_conditions,
)
from torchdyno.matrices import Dense, Diagonal, SkewCoupling, normal
from torchdyno.objectives import SequenceClassification
from torchdyno.readout import Last
from torchdyno.sequence import EulerCell, Gate, LeakyField, Recurrent

train, val, test = frequency_task()
data = BenchmarkDataset("frequency", SequenceClassification(), train, test, val)
H = 16 * 8


def make_model():
    return nn.Sequential(
        Recurrent(
            EulerCell(
                LeakyField(
                    recurrent=Diagonal(H, bound=torch.tanh),
                    coupling=SkewCoupling([16] * 8, pairs=8),
                    input=Dense(1, H, init=normal(H**-0.5)),
                    activation=torch.relu,
                ),
                step=Gate(1, H, bounds=(1e-5, 0.2)),
            )
        ),
        Last(),
        nn.Linear(H, 3),
    )


results = run_conditions(
    BenchmarkSpec(data, seeds=[0, 1]),
    make_model,
    l2=[1e-4, 1e-2, 1.0],
    lr=1e-2,
    epochs=15,
)
print(conditions_table(results))
