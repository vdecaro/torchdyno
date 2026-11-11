"""Backprop: gradient training of whatever requires grad."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import torch

from torchdyno.learners.base import (
    FitResult,
    drop_washout,
    evaluate,
    restore_modes,
)
from torchdyno.reproducibility import get_rng_state

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
    )

    from torch import (
        Tensor,
        nn,
    )


def _build_optimizer(
    optimizer: str | Callable[[Any], torch.optim.Optimizer],
    params: Any,
    lr: float,
) -> torch.optim.Optimizer:
    if callable(optimizer):
        return optimizer(params)
    if optimizer == "adam":
        return torch.optim.Adam(params, lr=lr)
    if optimizer == "sgd":
        return torch.optim.SGD(params, lr=lr)
    raise ValueError(
        f"Unknown optimizer: {optimizer!r}. Use 'adam', 'sgd', or a callable."
    )


class Backprop:
    """Train every parameter of a model that requires grad.

    Parts frozen with ``requires_grad_(False)`` stay bit-identical while the rest
    trains. The model trains in train mode and ends in the modes it started in.
    """

    def __init__(
        self,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer: str | Callable[[Any], torch.optim.Optimizer] = "adam",
        lr: float = 1e-3,
        epochs: int = 10,
        scheduler: Callable[[torch.optim.Optimizer], Any] | None = None,
        score_fn: Callable[[Tensor, Tensor], float] | None = None,
        washout: int = 0,
    ):
        """Initializes the trainer.

        Args:
            loss_fn: ``loss_fn(prediction, target) -> Tensor`` (e.g.
                ``nn.CrossEntropyLoss()`` or ``nn.MSELoss()``).
            optimizer: ``"adam"``/``"sgd"`` or a callable ``params -> Optimizer``.
            lr: learning rate (used only for the string optimizers).
            epochs: number of passes over ``train``.
            scheduler: optional ``optimizer -> LRScheduler``, stepped once per
                epoch.
            score_fn: optional ``score_fn(prediction, target) -> float``
                evaluated on ``val`` each epoch, recorded in
                ``history["val_score"]``.
            washout: leading timesteps of time-aligned predictions left out of
                the loss and the validation score.
        """
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.scheduler = scheduler
        self.score_fn = score_fn
        self.washout = washout

    def fit(
        self,
        model: nn.Module,
        train: Iterable[tuple[Tensor, Tensor]],
        val: Iterable[tuple[Tensor, Tensor]] | None = None,
    ) -> FitResult:
        """Train ``model`` on ``train``.

        Args:
            model: the model to train in place.
            train: batches of ``(x, y)``, iterated once per epoch.
            val: optional batches scored with ``score_fn`` after each epoch.

        Returns:
            The per-epoch history and the RNG state at the start.
        """
        rng = get_rng_state()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = _build_optimizer(self.optimizer, params, self.lr)
        sched = self.scheduler(opt) if self.scheduler is not None else None
        history: dict[str, list[float]] = {"train_loss": []}

        with restore_modes(model):
            for _ in range(self.epochs):
                model.train()
                total, n = 0.0, 0
                for x, y in train:
                    opt.zero_grad()
                    loss = self.loss_fn(*drop_washout(model(x), y, self.washout))
                    loss.backward()
                    opt.step()
                    total += loss.item()
                    n += 1
                history["train_loss"].append(total / max(n, 1))
                if sched is not None:
                    sched.step()
                if val is not None and self.score_fn is not None:
                    history.setdefault("val_score", []).append(
                        evaluate(model, val, self.score_fn, self.washout)
                    )

        return FitResult(history=history, best={"epochs": self.epochs}, rng=rng)
