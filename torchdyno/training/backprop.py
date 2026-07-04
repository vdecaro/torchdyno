"""BackpropTrainer: standard gradient training as an external Learner.

A loop-owning trainer for differentiable cores (e.g. AdaDiag/SCN). It steps a
``torch.optim`` optimizer over ``model.parameters()`` — the model owns no
training logic; the trainer is attached from outside.
"""

from typing import (
    Any,
    Callable,
    Optional,
    Union,
)

import torch
from torch import nn

from torchdyno.training.base import FitResult


def _build_optimizer(
    optimizer: Union[str, Callable[[Any], torch.optim.Optimizer]],
    params: Any,
    lr: float,
) -> torch.optim.Optimizer:
    if callable(optimizer):
        return optimizer(params)
    if optimizer == "adam":
        return torch.optim.Adam(params, lr=lr)
    if optimizer == "sgd":
        return torch.optim.SGD(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {optimizer!r}. Use 'adam', 'sgd', or a callable.")


class BackpropTrainer:
    """Train a model by backpropagation over its parameters.

    Args:
        loss_fn: ``loss_fn(prediction, target) -> Tensor`` (e.g.
            ``nn.CrossEntropyLoss()`` or ``nn.MSELoss()``).
        optimizer: ``"adam"``/``"sgd"`` or a callable ``params -> Optimizer``.
        lr: learning rate (used only for the string optimizers).
        epochs: number of passes over ``train``.
        score_fn: optional ``score_fn(prediction, target) -> float`` evaluated on
            ``val`` each epoch, recorded in ``history["val_score"]``.
    """

    def __init__(
        self,
        loss_fn: Callable[[Any, Any], torch.Tensor],
        optimizer: Union[str, Callable[[Any], torch.optim.Optimizer]] = "adam",
        lr: float = 1e-3,
        epochs: int = 10,
        score_fn: Optional[Callable[[Any, Any], float]] = None,
    ):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.score_fn = score_fn

    def fit(self, model: nn.Module, train: Any, val: Optional[Any] = None) -> FitResult:
        opt = _build_optimizer(self.optimizer, model.parameters(), self.lr)
        history: dict = {"train_loss": []}
        if val is not None and self.score_fn is not None:
            history["val_score"] = []

        for _ in range(self.epochs):
            model.train()
            total, n = 0.0, 0
            for x, y in train:
                opt.zero_grad()
                loss = self.loss_fn(model(x), y)
                loss.backward()
                opt.step()
                total += loss.item()
                n += 1
            history["train_loss"].append(total / max(n, 1))

            if val is not None and self.score_fn is not None:
                model.eval()
                scores, m = 0.0, 0
                with torch.no_grad():
                    for x, y in val:
                        scores += self.score_fn(model(x), y)
                        m += 1
                history["val_score"].append(scores / max(m, 1))

        return FitResult(history=history, best={"epochs": self.epochs}, extras={})
