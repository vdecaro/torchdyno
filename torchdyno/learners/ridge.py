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
from torchdyno.learners.scoring import evaluate
from torchdyno.reproducibility import get_rng_state

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


class Ridge:
    """Fit a linear readout on whatever reaches it, in closed form.

    The readout is ``model[-1]`` by default, or the real ``nn.Linear`` or
    :class:`~torchdyno.matrices.Dense` that ``readout`` names, wherever it sits:
    a readout in the middle of a model is fitted on its own input, against
    targets for its own output. Time-aligned ``(T, B, H)`` features are fitted
    at every timestep, pooled ``(B, H)`` ones once per sequence; integer targets
    become one-hot. Only the readout changes. The sums are kept in float64, and
    when there are fewer samples than features the solve uses the dual form.
    """

    def __init__(
        self,
        l2: float | Sequence[float] | Tensor = 1e-6,
        washout: int = 0,
        standardize: bool = False,
        score_fn: Callable[[Tensor, Tensor], float] | None = None,
        mode: Literal["min", "max"] | None = None,
        readout: str | None = None,
    ):
        """Initializes the learner.

        Args:
            l2: the L2 penalty, or candidates to select among on ``val``
                (requires ``score_fn``): a sequence or a tensor.
            washout: leading timesteps to discard, when fitting and when
                scoring ``val``; only for time-aligned features.
            standardize: scale every feature to unit variance before solving,
                folding the scale back into the weight.
            score_fn: ``score_fn(prediction, target) -> float`` ranking the L2
                candidates.
            mode: ``"min"`` or ``"max"``: whether the best score is smallest or
                largest; ``None`` takes ``score_fn.mode``, as a
                :class:`~torchdyno.objectives.Metric` has.
            readout: the readout's name in the model, as
                ``model.get_submodule`` takes it; ``None`` takes ``model[-1]``.

        Raises:
            ValueError: if there is no ``l2`` value, one is negative, or ``mode``
                is neither ``"min"`` nor ``"max"``.
        """
        self.l2: list[float] = (
            torch.as_tensor(l2, dtype=torch.float64).flatten().tolist()
        )
        if not self.l2 or min(self.l2) < 0:
            raise ValueError(f"l2 needs at least one value, all >= 0, got {l2}.")
        if mode not in {"min", "max", None}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}.")
        mode = mode or getattr(score_fn, "mode", None)
        self.washout = washout
        self.standardize = standardize
        self.score_fn = score_fn
        self.mode = mode
        self.readout = readout

    def fit(
        self,
        model: nn.Module,
        train: Iterable[tuple[Tensor, Tensor]],
        val: Iterable[tuple[Tensor, Tensor]] | None = None,
    ) -> FitResult:
        """Solve for the readout on ``train``.

        The readout gets an intercept if it has a bias.

        Args:
            model: the model whose readout is set in place.
            train: batches of time-first ``(x, y)``; ``y`` is ``(T, B, O)``
                for time-aligned features or ``(B, O)`` / ``(B,)`` class
                indices for pooled ones.
            val: batches used to select among several ``l2`` values.

        Returns:
            The chosen ``l2`` (and its score when selecting), the score of every
            candidate in ``history`` when selecting, and the RNG state.

        Raises:
            TypeError: if the readout is not a real ``nn.Linear`` or ``Dense``.
            ValueError: if several ``l2`` values are given without ``val``,
                ``score_fn`` and a ``mode``.
        """
        readout = find_readout(model, self.readout)
        rng = get_rng_state()
        candidates = self.l2
        if len(candidates) > 1 and (
            val is None or self.score_fn is None or not self.mode
        ):
            raise ValueError(
                "Selecting among several l2 values needs val, score_fn and mode."
            )

        solve = self._solver(model, readout, train)
        history: dict[str, list[float]] = {}
        if len(candidates) == 1:
            best: dict[str, float] = {"l2": candidates[0]}
        else:
            assert val is not None and self.score_fn is not None  # checked above
            scores = {}
            for l2 in candidates:
                _set(readout, *solve(l2))
                scores[l2] = evaluate(model, val, self.score_fn, self.washout)
            pick = max if self.mode == "max" else min
            chosen = pick(scores, key=scores.__getitem__)
            best = {"l2": chosen, "score": scores[chosen]}
            history = {"l2": list(scores), "val_score": list(scores.values())}
        _set(readout, *solve(best["l2"]))
        return FitResult(history=history, best=best, rng=rng)

    def _solver(
        self,
        model: nn.Module,
        readout: nn.Linear | Dense,
        loader: Iterable[tuple[Tensor, Tensor]],
    ) -> Callable[[float], tuple[Tensor, Tensor]]:
        """Accumulate the data once; return ``l2 -> (weight, bias)``."""
        xs: list[Tensor] = []
        ys: list[Tensor] = []
        gram = cross = sum_x = sum_y = None
        n = 0
        for x, y in loader:
            f, t = feature_rows(model, readout, x, y, self.washout)
            n += f.shape[0]
            if gram is None:
                xs.append(f)
                ys.append(t)
                if n < f.shape[1]:  # fewer samples than features: keep the rows
                    continue
                f, t = torch.cat(xs), torch.cat(ys)
                xs, ys = [], []
                gram, cross, sum_x, sum_y = f.T @ f, f.T @ t, f.sum(0), t.sum(0)
            else:
                assert cross is not None and sum_x is not None and sum_y is not None
                gram, cross = gram + f.T @ f, cross + f.T @ t
                sum_x, sum_y = sum_x + f.sum(0), sum_y + t.sum(0)

        if n == 0:
            raise ValueError("train has no samples to fit on.")
        intercept = readout.bias is not None
        if gram is None:  # dual form on the kept rows
            rows, targets = torch.cat(xs), torch.cat(ys)
            mean_x, mean_y = rows.mean(0), targets.mean(0)
            centered = rows - mean_x
            scale = _scale(centered.pow(2).mean(0), self.standardize)
            if intercept:
                rows, targets = centered, targets - mean_y
            rows = rows / scale
            kernel = rows @ rows.T
            eye = torch.eye(n, dtype=kernel.dtype, device=kernel.device)

            def solve(l2: float) -> tuple[Tensor, Tensor]:
                coef = rows.T @ torch.linalg.solve(kernel + l2 * eye, targets)
                return _unscale(coef, scale, mean_x, mean_y, intercept)

            return solve

        assert cross is not None and sum_x is not None and sum_y is not None
        mean_x, mean_y = sum_x / n, sum_y / n
        gram_c = gram - n * torch.outer(mean_x, mean_x)
        scale = _scale(gram_c.diagonal() / n, self.standardize)
        if intercept:
            gram, cross = gram_c, cross - n * torch.outer(mean_x, mean_y)
        gram = gram / torch.outer(scale, scale)
        cross = cross / scale[:, None]
        eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)

        def solve_primal(l2: float) -> tuple[Tensor, Tensor]:
            coef = torch.linalg.solve(gram + l2 * eye, cross)
            return _unscale(coef, scale, mean_x, mean_y, intercept)

        return solve_primal


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
