import torch

import torchdyno
from torchdyno.heads import ClassificationHead
from torchdyno.model import SequenceModel
from torchdyno.models.assembly.adadiag import AdaDiagCore
from torchdyno.training.backprop import BackpropTrainer
from torchdyno.training.base import FitResult, Learner


def _classification_data(T=30, B=16, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(T, B, 1)
    y = (x.mean(dim=0).squeeze(-1) > 0).long()  # (B,)
    return [(x, y)]


def _model(seed=0):
    torch.manual_seed(seed)
    core = AdaDiagCore(input_size=1, block_sizes=[16, 16], coupling_topology=[(0, 1)])
    return SequenceModel(core, ClassificationHead(core.state_size, 2, pool="last"))


def test_is_learner():
    assert isinstance(BackpropTrainer(loss_fn=torch.nn.CrossEntropyLoss()), Learner)


def test_exported_from_training():
    assert hasattr(torchdyno.training, "BackpropTrainer")


def test_fit_reduces_loss():
    model = _model()
    train = _classification_data()
    trainer = BackpropTrainer(
        loss_fn=torch.nn.CrossEntropyLoss(), optimizer="adam", lr=1e-2, epochs=60
    )
    result = trainer.fit(model, train)
    assert isinstance(result, FitResult)
    losses = result.history["train_loss"]
    assert len(losses) == 60
    assert losses[-1] < losses[0]


def test_records_val_score():
    model = _model()
    train = _classification_data(seed=0)
    val = _classification_data(seed=1)

    def accuracy(logits, y):
        return (logits.argmax(dim=-1) == y).float().mean().item()

    trainer = BackpropTrainer(
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer="adam",
        lr=1e-2,
        epochs=5,
        score_fn=accuracy,
    )
    result = trainer.fit(model, train, val)
    assert len(result.history["val_score"]) == 5


def test_accepts_optimizer_callable():
    model = _model()
    trainer = BackpropTrainer(
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=lambda params: torch.optim.SGD(params, lr=1e-2),
        epochs=3,
    )
    result = trainer.fit(model, _classification_data())
    assert len(result.history["train_loss"]) == 3


def test_score_fn_receives_pred_then_target():
    model = _model()
    train = _classification_data(seed=0)
    val = _classification_data(seed=1)
    seen = {}

    def probe(pred, target):
        seen["pred_ndim"] = pred.ndim
        seen["target_ndim"] = target.ndim
        return 0.0

    BackpropTrainer(
        loss_fn=torch.nn.CrossEntropyLoss(), epochs=1, score_fn=probe
    ).fit(model, train, val)
    assert seen["pred_ndim"] == 2 and seen["target_ndim"] == 1
