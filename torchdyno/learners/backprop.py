"""Backprop: gradient training of whatever requires grad."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
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
    params: list[tuple[str, nn.Parameter]],
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
        clip: float | None = None,
        schedule: Literal["epoch", "batch"] = "epoch",
        callback: Callable[[int, dict[str, list[float]], nn.Module], Any] | None = None,
        keep_best: bool = False,
        mode: Literal["min", "max"] | None = None,
    ):
        """Initializes the trainer.

        Args:
            loss_fn: ``loss_fn(prediction, target) -> Tensor`` (e.g.
                ``nn.CrossEntropyLoss()`` or ``nn.MSELoss()``).
            optimizer: ``"adam"``/``"sgd"``, or a callable that builds the
                optimizer from the ``(name, parameter)`` pairs that require grad,
                so it can group them by name, e.g. without weight decay on the
                transitions.
            lr: learning rate (used only for the string optimizers).
            epochs: number of passes over ``train``.
            scheduler: optional ``optimizer -> LRScheduler``, stepped as
                ``schedule`` says; a ``ReduceLROnPlateau`` is stepped every epoch
                on the validation score, or on the training loss without one.
            score_fn: optional ``score_fn(prediction, target) -> float``
                evaluated on ``val`` each epoch, recorded in
                ``history["val_score"]``.
            washout: leading timesteps of time-aligned predictions left out of
                the loss and the validation score.
            clip: the largest gradient norm, or ``None`` not to clip.
            schedule: step the scheduler after every ``"epoch"`` or ``"batch"``.
            callback: called as ``callback(epoch, history, model)`` after every
                epoch, e.g. to log; returning ``True`` stops training.
            keep_best: end with the parameters of the epoch with the best
                validation score, rather than the last.
            mode: ``"min"`` or ``"max"``: whether the best validation score is
                smallest or largest; ``None`` takes ``score_fn.mode``.

        Raises:
            ValueError: if ``schedule`` is unknown, or ``keep_best`` has no
                ``score_fn`` and ``mode`` to rank epochs by.
        """
        if schedule not in {"epoch", "batch"}:
            raise ValueError(f"schedule must be 'epoch' or 'batch', got {schedule!r}.")
        mode = mode or getattr(score_fn, "mode", None)
        if keep_best and (score_fn is None or mode not in {"min", "max"}):
            raise ValueError("keep_best needs a score_fn and a mode to rank epochs.")
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.scheduler = scheduler
        self.score_fn = score_fn
        self.washout = washout
        self.clip = clip
        self.schedule = schedule
        self.callback = callback
        self.keep_best = keep_best
        self.mode = mode

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
            The per-epoch history, the epochs run and, with a validation score,
            the best epoch and its score; and the RNG state at the start.

        Raises:
            ValueError: if ``train`` has no batches, or ``keep_best`` has no
                ``val``.
        """
        if self.keep_best and val is None:
            raise ValueError("keep_best needs val to rank epochs on.")
        rng = get_rng_state()
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        opt = _build_optimizer(self.optimizer, params, self.lr)
        sched = self.scheduler(opt) if self.scheduler is not None else None
        history: dict[str, list[float]] = {"train_loss": []}
        best: dict[str, Any] = {}
        kept: dict[str, Tensor] | None = None

        with restore_modes(model):
            for epoch in range(self.epochs):
                history["train_loss"].append(self._epoch(model, train, opt, sched))
                if val is not None and self.score_fn is not None:
                    score = evaluate(model, val, self.score_fn, self.washout)
                    history.setdefault("val_score", []).append(score)
                    if not best or self._better(score, best["score"]):
                        best = {"epoch": epoch, "score": score}
                        if self.keep_best:
                            kept = {k: v.clone() for k, v in model.state_dict().items()}
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sched.step(history.get("val_score", history["train_loss"])[-1])
                elif sched is not None and self.schedule == "epoch":
                    sched.step()
                if self.callback is not None and self.callback(epoch, history, model):
                    break
        if kept is not None:
            model.load_state_dict(kept)
        best["epochs"] = len(history["train_loss"])
        return FitResult(history=history, best=best, rng=rng)

    def _epoch(
        self,
        model: nn.Module,
        train: Iterable[tuple[Tensor, Tensor]],
        opt: torch.optim.Optimizer,
        sched: Any,
    ) -> float:
        """One pass over ``train``; the mean training loss."""
        model.train()
        total, n = 0.0, 0
        for x, y in train:
            opt.zero_grad()
            loss = self.loss_fn(*drop_washout(model(x), y, self.washout))
            loss.backward()
            if self.clip is not None:
                params = [p for group in opt.param_groups for p in group["params"]]
                torch.nn.utils.clip_grad_norm_(params, self.clip)
            opt.step()
            if self.schedule == "batch" and sched is not None:
                sched.step()
            total += loss.item()
            n += 1
        if not n:
            raise ValueError("train has no batches to train on.")
        return total / n

    def _better(self, score: float, best: float) -> bool:
        return score > best if self.mode == "max" else score < best
