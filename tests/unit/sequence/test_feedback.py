import torch
from torch import nn

from tests.unit.compositions import reservoir
from torchdyno.matrices import Dense
from torchdyno.sequence import (
    FeedbackCell,
    Recurrent,
)
from torchdyno.testing import assert_layer_conforms


def closed_loop(features=1, size=8, outputs=2):
    inner = reservoir(features + outputs, size).cell
    return FeedbackCell(inner, nn.Linear(size, outputs))


def test_feedback_cell_reads_its_readout_as_extra_input():
    torch.manual_seed(0)
    cell = closed_loop()
    u, h = torch.randn(3, 1), torch.randn(3, 8)
    expected = cell.cell(torch.cat([u, cell.readout(h)], -1), h)
    torch.testing.assert_close(cell(u, h), expected)
    assert_layer_conforms(Recurrent(cell), torch.randn(6, 3, 1))


def test_a_closed_loop_runs_on_an_input_without_features():
    torch.manual_seed(0)
    cell = FeedbackCell(reservoir(2, 8).cell, Dense(8, 2))
    out = Recurrent(cell)(torch.empty(5, 3, 0), h0=torch.randn(3, 8))
    assert out.shape == (5, 3, 8)
    assert out.isfinite().all()
