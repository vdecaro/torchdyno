"""Ridge: fit a linear readout of a model in closed form."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
)

import torch

from torchdyno.learners.features import (
    feature_rows,
    find_readout,
)
from torchdyno.learners.protocol import FitResult
from torchdyno.learners.scoring import (
    drop_washout,
    evaluate,
    restore_modes,
    score_outputs,
)
from torchdyno.reproducibility import get_rng_state
from torchdyno.sequence._checks import _absent

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
        Sequence,
    )

    from torch import (
        Tensor,
        nn,
    )

    from torchdyno.matrices import Dense

    Solve = Callable[[float], tuple[Tensor, Tensor]]


class Ridge:
    """Fit a linear readout on whatever reaches it, in closed form.

    The readout is ``model[-1]`` by default, or the real ``nn.Linear`` or
    [`Dense`][torchdyno.matrices.Dense] that ``readout`` names, wherever it sits:
    a readout in the middle of a model is fitted on its own input, against
    targets for its own output. Time-aligned ``(T, B, H)`` features are fitted
    at every timestep, pooled ``(B, H)`` ones once per sequence; integer targets
    become one-hot. Only the readout changes.

    The sums are kept in float64. With at least as many samples as features
    they are accumulated batch by batch into an ``(H, H)`` matrix solved in
    place by Cholesky; with fewer, or for leave-one-out selection, the rows are
    kept and solved by one SVD for every candidate penalty.
    """

    def __init__(
        self,
        l2: float | Sequence[float] | Tensor = 1e-6,
        washout: int = 0,
        standardize: bool = False,
        score_fn: Callable[[Tensor, Tensor], float] | None = None,
        mode: Literal["min", "max"] | None = None,
        readout: str | None = None,
        class_weight: Literal["balanced"] | Sequence[float] | Tensor | None = None,
    ):
        """Initializes the learner.

        Args:
            l2: the L2 penalty, or candidates to select among (requires
                ``score_fn``): a sequence or a tensor. They are scored on
                ``val``, or by leave-one-out without it.
            washout: leading timesteps to discard, when fitting and when
                scoring ``val``; only for time-aligned features.
            standardize: scale every feature to unit variance before solving,
                folding the scale back into the weight.
            score_fn: ``score_fn(prediction, target) -> float`` ranking the L2
                candidates.
            mode: ``"min"`` or ``"max"``: whether the best score is smallest or
                largest; ``None`` takes ``score_fn.mode``, as a
                [`Metric`][torchdyno.objectives.Metric] has.
            readout: the readout's name in the model, as
                ``model.get_submodule`` takes it; ``None`` takes ``model[-1]``.
            class_weight: for integer targets, a weight per class on every
                sample's squared error, or ``"balanced"`` for ``n / (C n_c)``,
                so rare classes count as much as common ones.

        Raises:
            ValueError: if there is no ``l2`` value, one is negative, ``mode``
                is neither ``"min"`` nor ``"max"``, or ``class_weight`` is a
                string other than ``"balanced"``.
        """
        self.l2: list[float] = (
            torch.as_tensor(l2, dtype=torch.float64).flatten().tolist()
        )
        if not self.l2 or min(self.l2) < 0:
            raise ValueError(f"l2 needs at least one value, all >= 0, got {l2}.")
        if mode not in {"min", "max", None}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}.")
        if isinstance(class_weight, str) and class_weight != "balanced":
            raise ValueError(f"class_weight must be 'balanced', got {class_weight!r}.")
        mode = mode or getattr(score_fn, "mode", None)
        self.washout = washout
        self.standardize = standardize
        self.score_fn = score_fn
        self.mode = mode
        self.readout = readout
        self.class_weight = class_weight

    def fit(
        self,
        model: nn.Module,
        train: Iterable[tuple[Tensor, Tensor]],
        val: Iterable[tuple[Tensor, Tensor]] | None = None,
    ) -> FitResult:
        """Solve for the readout on ``train``.

        The readout gets an intercept if it has a bias, which is not penalised.

        Args:
            model: the model whose readout is set in place.
            train: batches of time-first ``(x, y)``; ``y`` is ``(T, B, O)``
                for time-aligned features or ``(B, O)`` / ``(B,)`` class
                indices for pooled ones.
            val: batches to select among several ``l2`` values on; without it
                they are selected by leave-one-out on ``train``, which keeps
                every row and, for time-aligned features, scores each timestep
                with its neighbours in the fit.

        Returns:
            The chosen ``l2`` (and its score when selecting), the score of every
            candidate in ``history`` (``val_score`` or ``loo_score``) when
            selecting, and the RNG state.

        Raises:
            TypeError: if the readout is not a real ``nn.Linear`` or ``Dense``.
            ValueError: if several ``l2`` values are given without ``score_fn``
                and a ``mode``, ``class_weight`` is set for float targets, or
                ``train`` has no samples.
        """
        readout = find_readout(model, self.readout)
        rng = get_rng_state()
        candidates = self.l2
        if len(candidates) > 1 and (self.score_fn is None or not self.mode):
            raise ValueError(
                "Selecting among several l2 values needs score_fn and mode."
            )
        loo = len(candidates) > 1 and val is None
        rows = _Rows(self, model, readout, train, keep=loo)
        solve = rows.solver()
        history: dict[str, list[float]] = {}
        best: dict[str, float] = {"l2": candidates[0]}
        if len(candidates) > 1:
            assert self.score_fn is not None  # checked above
            if loo:
                scores = {l2: self.score_fn(*rows.loo(l2)) for l2 in candidates}
            else:
                assert val is not None
                score = self._scorer(model, readout, val)
                scores = {}
                for l2 in candidates:
                    _set(readout, *solve(l2))
                    scores[l2] = score()
            pick = max if self.mode == "max" else min
            chosen = pick(scores, key=scores.__getitem__)
            best = {"l2": chosen, "score": scores[chosen]}
            key = "loo_score" if loo else "val_score"
            history = {"l2": list(scores), key: list(scores.values())}
        _set(readout, *solve(best["l2"]))
        return FitResult(history=history, best=best, rng=rng)

    def _scorer(
        self,
        model: nn.Module,
        readout: nn.Linear | Dense,
        val: Iterable[tuple[Tensor, Tensor]],
    ) -> Callable[[], float]:
        """Score the model on ``val`` with the readout as set.

        When the readout's output is the model's, its input is computed once and
        only the readout reruns for every candidate.
        """
        assert self.score_fn is not None
        score_fn, washout = self.score_fn, self.washout
        captured: list[tuple[Tensor, Tensor]] = []
        handle = readout.register_forward_hook(
            lambda _, args, out: captured.append((args[0], out))
        )
        inputs: list[Tensor] = []
        targets: list[Tensor] = []
        last = True
        try:
            with torch.no_grad(), restore_modes(model):
                model.eval()
                for x, y in val:
                    out = model(x)
                    last = len(captured) == 1 and captured[0][1] is out
                    if not last:
                        break
                    feature, target = drop_washout(captured.pop()[0], y, washout)
                    inputs.append(feature)
                    targets.append(target)
        finally:
            handle.remove()
        if not last:  # layers after the readout: rerun the whole model
            return lambda: evaluate(model, val, score_fn, washout)
        if not targets:
            raise ValueError("The loader has no batches to score.")

        def score() -> float:
            with torch.no_grad():
                preds = [readout(feature) for feature in inputs]
            return score_outputs(preds, targets, score_fn)

        return score


class _Rows:
    """The fitting data of a ridge readout, as sums or as kept rows."""

    def __init__(
        self,
        ridge: Ridge,
        model: nn.Module,
        readout: nn.Linear | Dense,
        loader: Iterable[tuple[Tensor, Tensor]],
        keep: bool,
    ):
        self.standardize = ridge.standardize
        self.intercept = readout.bias is not None
        class_weight = ridge.class_weight
        if class_weight == "balanced":
            class_weight = _balanced(loader, ridge.washout, readout.out_features)
        weights = None if class_weight is None else torch.as_tensor(class_weight)
        xs: list[Tensor] = []
        ys: list[Tensor] = []
        ws: list[Tensor] = []
        self.gram: Tensor | None = None
        self.integer = False
        for x, y in loader:
            self.integer = not y.is_floating_point()
            f, t = feature_rows(model, readout, x, y, ridge.washout)
            if weights is None:
                w = t.new_ones(len(t))
            elif self.integer:
                w = weights.to(t)[t.argmax(-1)]
            else:
                raise ValueError("class_weight needs integer class targets.")
            if self.gram is None:
                xs.append(f)
                ys.append(t)
                ws.append(w)
                if keep or sum(map(len, xs)) < f.shape[1]:
                    continue
                f, t, w = torch.cat(xs), torch.cat(ys), torch.cat(ws)
                xs, ys, ws = [], [], []
                self.gram, self.cross = f.T @ (w[:, None] * f), f.T @ (w[:, None] * t)
                self.sum_x, self.sum_y = w @ f, w @ t
                self.total = w.sum()
            else:
                self.gram.addmm_(f.T, w[:, None] * f)
                self.cross.addmm_(f.T, w[:, None] * t)
                self.sum_x += w @ f
                self.sum_y += w @ t
                self.total += w.sum()
        if self.gram is None and not xs:
            raise ValueError("train has no samples to fit on.")
        if self.gram is None:
            self.rows, self.targets = torch.cat(xs), torch.cat(ys)
            self.weights = torch.cat(ws)

    def solver(self) -> Solve:
        """``l2 -> (weight, bias)``."""
        return self._from_rows() if self.gram is None else self._from_sums()

    def _from_sums(self) -> Solve:
        """Solve the normal equations, centred and scaled in place."""
        assert self.gram is not None
        gram, cross, total = self.gram, self.cross, float(self.total)
        mean_x, mean_y = self.sum_x / total, self.sum_y / total
        scale = _scale((gram.diagonal() - total * mean_x**2) / total, self.standardize)
        if self.intercept:
            gram.addr_(mean_x, mean_x, alpha=-total)
            cross.addr_(mean_x, mean_y, alpha=-total)
        gram.div_(scale[:, None]).div_(scale)
        cross.div_(scale[:, None])
        diagonal = gram.diagonal().clone()

        def solve(l2: float) -> tuple[Tensor, Tensor]:
            gram.diagonal().add_(l2)
            try:
                factor, info = torch.linalg.cholesky_ex(gram)
                if info:  # not positive definite, e.g. collinear features and l2 = 0
                    coef = torch.linalg.solve(gram, cross)
                else:
                    coef = torch.cholesky_solve(cross, factor)
            finally:
                gram.diagonal().copy_(diagonal)
            return _unscale(coef, scale, mean_x, mean_y, self.intercept)

        return solve

    def _from_rows(self) -> Solve:
        """Solve on the kept rows by one SVD, kept for leave-one-out scoring."""
        rows, targets, weights = self.rows, self.targets, self.weights
        total = weights.sum()
        mean_x, mean_y = weights @ rows / total, weights @ targets / total
        centered = rows - mean_x
        scale = _scale(weights @ centered**2 / total, self.standardize)
        if self.intercept:
            rows, targets = centered, targets - mean_y
        root = weights.sqrt()[:, None]
        u, s, vh = torch.linalg.svd(root * rows / scale, full_matrices=False)
        weighted = root * targets
        projected = u.T @ weighted
        self.svd = (u, s, projected, weighted, root, total)

        def solve(l2: float) -> tuple[Tensor, Tensor]:
            coef = vh.T @ ((s / (s**2 + l2))[:, None] * projected)
            return _unscale(coef, scale, mean_x, mean_y, self.intercept)

        return solve

    def loo(self, l2: float) -> tuple[Tensor, Tensor]:
        """Every row's prediction by the fit without it, and the targets to score.

        Ridge is a linear smoother, so the left-out residual is the fitted
        residual over ``1 - h``, with ``h`` the row's leverage; the unpenalised
        intercept adds ``w_i / Σ w`` to it.
        """
        u, s, projected, weighted, root, total = self.svd
        shrink = s**2 / (s**2 + l2)
        residual = (weighted - u @ (shrink[:, None] * projected)) / root
        leverage = (u**2) @ shrink
        if self.intercept:
            leverage = leverage + root[:, 0] ** 2 / total
        pred = self.targets - residual / (1 - leverage)[:, None]
        return pred, self.targets.argmax(-1) if self.integer else self.targets


def _balanced(
    loader: Iterable[tuple[Tensor, Tensor]], washout: int, classes: int
) -> Tensor:
    """``n / (C n_c)`` from the labels of every present, kept step."""
    counts = torch.zeros(classes, dtype=torch.float64)
    for x, y in loader:
        if y.is_floating_point():
            raise ValueError("class_weight needs integer class targets.")
        labels = y[washout:][~_absent(x)[washout:]] if y.ndim == 2 else y  # noqa: PLR2004
        counts += torch.bincount(labels.flatten(), minlength=classes).double().cpu()
    return torch.where(counts > 0, counts.sum() / (classes * counts), 0)


def _scale(variance: Tensor, standardize: bool) -> Tensor:
    """Per-feature standard deviation, or ones; zero-variance features keep 1."""
    if not standardize:
        return torch.ones_like(variance)
    std = variance.clamp_min(0).sqrt()
    return torch.where(std > 0, std, torch.ones_like(std))


def _unscale(
    coef: Tensor, scale: Tensor, mean_x: Tensor, mean_y: Tensor, intercept: bool
) -> tuple[Tensor, Tensor]:
    """Map a solution on scaled features back to ``(weight, bias)`` on raw ones."""
    weight = coef / scale[:, None]  # (H, O)
    bias = mean_y - mean_x @ weight if intercept else torch.zeros_like(mean_y)
    return weight.T, bias


def _set(readout: nn.Linear | Dense, weight: Tensor, bias: Tensor) -> None:
    with torch.no_grad():
        readout.weight.copy_(weight)
        if readout.bias is not None:
            readout.bias.copy_(bias)
