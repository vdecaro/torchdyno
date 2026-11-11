import pytest
import torch
from torch import nn

from tests.unit.compositions import reservoir
from tests.unit.sequence.test_feedback import closed_loop
from torchdyno.learners import (
    FORCE,
    RLS,
)
from torchdyno.sequence import (
    FeedbackCell,
    Recurrent,
)

DTYPE = torch.float64


def test_updates_the_readout_at_every_step_of_the_running_loop():
    torch.manual_seed(0)
    cell = closed_loop().to(DTYPE)
    model = nn.Sequential(Recurrent(cell), cell.readout)
    x, y = torch.randn(12, 2, 1, dtype=DTYPE), torch.randn(12, 2, 2, dtype=DTYPE)
    x[10:, 1] = torch.nan  # the second sequence is shorter
    readout = nn.Linear(8, 2, dtype=DTYPE)
    readout.load_state_dict(cell.readout.state_dict())
    expected, rls = FeedbackCell(cell.cell, readout), RLS(l2=0.5)
    h = torch.zeros(2, 8, dtype=DTYPE)
    with torch.no_grad():
        for t in range(12):
            present = ~x[t, :, 0].isnan()
            h = torch.where(present[:, None], expected(x[t].nan_to_num(), h), h)
            if t >= 3:
                rls.update(readout, h[present], y[t][present])
    FORCE(l2=0.5, washout=3).fit(model, [(x, y)])
    torch.testing.assert_close(cell.readout.weight, readout.weight)
    torch.testing.assert_close(cell.readout.bias, readout.bias)


def test_needs_one_feedback_cell_and_matching_targets():
    model = nn.Sequential(reservoir(1, 8), nn.Linear(8, 1))
    with pytest.raises(TypeError, match="one FeedbackCell"):
        FORCE().fit(model, [(torch.randn(4, 1, 1), torch.randn(4, 1, 1))])
    cell = closed_loop()
    with pytest.raises(ValueError, match="y must be"):
        FORCE().fit(Recurrent(cell), [(torch.randn(4, 1, 1), torch.randn(4, 1, 1))])
