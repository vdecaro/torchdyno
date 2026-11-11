import pytest
import torch
from torch import nn

from tests.unit.compositions import adadiag
from torchdyno.learners import Backprop
from torchdyno.pooling import Last


def _data(seed=0):
    torch.manual_seed(seed)
    x = torch.randn(20, 16, 2)
    return [(x, (x[..., 0].mean(dim=0) > 0).long())]


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(2, 2), adadiag(features=2), Last(), nn.Linear(12, 2))


def _snapshot(module):
    return {k: v.clone() for k, v in module.state_dict().items()}


def _changed(before, module):
    return {k for k, v in module.state_dict().items() if not torch.equal(before[k], v)}


def test_frozen_parts_stay_bit_identical_while_the_rest_trains():
    model = _model()
    model[1].requires_grad_(False)
    before = [_snapshot(m) for m in model]
    Backprop(nn.CrossEntropyLoss(), lr=1e-2, epochs=5).fit(model, _data())
    assert not _changed(before[1], model[1])
    assert _changed(before[0], model[0]) == {"weight", "bias"}
    assert _changed(before[3], model[3]) == {"weight", "bias"}


def test_trains_everything_that_requires_grad():
    model = _model()
    before = _snapshot(model[1])
    Backprop(nn.CrossEntropyLoss(), lr=1e-2, epochs=5).fit(model, _data())
    assert _changed(before, model[1])


def test_fit_reduces_loss():
    result = Backprop(nn.CrossEntropyLoss(), lr=1e-2, epochs=40).fit(_model(), _data())
    losses = result.history["train_loss"]
    assert len(losses) == 40
    assert losses[-1] < losses[0]
    assert result.rng is not None


def test_scheduler_steps_once_per_epoch():
    seen = []

    def scheduler(opt):
        seen.append(opt)
        return torch.optim.lr_scheduler.MultiStepLR(opt, [2], 0.1)

    Backprop(nn.CrossEntropyLoss(), lr=1.0, epochs=3, scheduler=scheduler).fit(
        _model(), _data()
    )
    assert seen[0].param_groups[0]["lr"] == pytest.approx(0.1)


def test_records_val_score_per_epoch():
    def accuracy(logits, y):
        return (logits.argmax(dim=-1) == y).float().mean().item()

    trainer = Backprop(nn.CrossEntropyLoss(), epochs=3, score_fn=accuracy)
    assert len(trainer.fit(_model(), _data(0), _data(1)).history["val_score"]) == 3


def test_accepts_optimizer_callable_and_rejects_unknown_names():
    trainer = Backprop(
        nn.CrossEntropyLoss(), optimizer=lambda p: torch.optim.SGD(p, lr=1e-2), epochs=2
    )
    assert len(trainer.fit(_model(), _data()).history["train_loss"]) == 2
    with pytest.raises(ValueError, match="Unknown optimizer"):
        Backprop(nn.CrossEntropyLoss(), optimizer="lbfgs").fit(_model(), _data())


def test_leaves_the_washout_out_of_the_loss_and_the_validation_score():
    seen = []

    def loss_fn(pred, target):
        seen.append(len(pred))
        return torch.mean((pred - target) ** 2)

    data = [(torch.randn(10, 4, 2), torch.randn(10, 4, 2))]
    trainer = Backprop(
        loss_fn, epochs=1, score_fn=lambda p, t: float(len(p)), washout=3
    )
    result = trainer.fit(nn.Sequential(nn.Linear(2, 2)), data, data)
    assert seen == [7]
    assert result.history["val_score"] == [7.0]
