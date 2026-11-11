from typing import TYPE_CHECKING

import torch
from torch import nn

from tests.unit.compositions import reservoir
from torchdyno.learners import (
    Backprop,
    Chain,
    IntrinsicPlasticity,
    Ridge,
)

if TYPE_CHECKING:
    from torchdyno.learners import Learner


def _data():
    torch.manual_seed(0)
    x = torch.randn(30, 4, 2)
    return [(x, torch.roll(x, 1, 0))]


def test_runs_each_learner_in_turn_on_the_same_model():
    torch.manual_seed(0)
    model = nn.Sequential(reservoir(), nn.Linear(16, 2))
    reference = nn.Sequential(reservoir(), nn.Linear(16, 2))
    reference.load_state_dict(model.state_dict())
    result = Chain(IntrinsicPlasticity(lr=0.1), Ridge(1e-3)).fit(model, _data())
    IntrinsicPlasticity(lr=0.1).fit(reference, _data())
    Ridge(1e-3).fit(reference, _data())
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value)
    assert result.best == {"1.l2": 1e-3}


def test_keys_every_learner_by_its_place():
    model = nn.Sequential(reservoir(), nn.Linear(16, 2))
    result = Chain(Ridge(1e-3), Backprop(nn.MSELoss(), epochs=2)).fit(model, _data())
    assert result.best["0.l2"] == 1e-3
    assert len(result.history["1.train_loss"]) == 2


def test_learners_and_chains_satisfy_the_protocol():
    learner: Learner = Chain(Ridge(), Chain(Ridge()))
    assert callable(learner.fit)
