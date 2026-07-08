"""RidgeRegression: closed-form readout fitting as an external optimizer.

Wraps the incremental ridge machinery in ``torchdyno.optim.ridge_regression``
behind the :class:`~torchdyno.training.base.Learner` surface, so a frozen-core
model (e.g. an ESN) is trained by ``RidgeRegression(...).fit(model, train)``
without the model owning any training logic.
"""

from typing import (
    Any,
    Callable,
    List,
    Literal,
    Optional,
    Union,
)

from torchdyno.optim.ridge_regression import (
    fit_and_validate_readout,
    fit_readout,
)
from torchdyno.registry import register_learner
from torchdyno.reproducibility import get_rng_state
from torchdyno.training.base import FitResult


@register_learner("ridge")
class RidgeRegression:
    """Fit a linear readout in closed form (optionally selecting L2 on a val set).

    Args:
        l2: L2 regularization value, or a list of candidates to select among
            (requires ``score_fn`` + ``mode`` + a ``val`` loader).
        washout: number of leading rows to skip when accumulating the ridge
            matrices (transient discard).
        score_fn: ``score_fn(pred, target) -> float`` used to rank L2 candidates
            on the validation set.
        mode: ``"min"`` or ``"max"`` — whether the best score is smallest or
            largest.
        store_matrices: if ``True``, keep the ridge ``A``/``B`` matrices in
            ``FitResult.extras``.
    """

    def __init__(
        self,
        l2: Union[float, List[float]] = 1e-6,
        washout: int = 0,
        score_fn: Optional[Callable[[Any, Any], float]] = None,
        mode: Optional[Literal["min", "max"]] = None,
        store_matrices: bool = False,
    ):
        self.l2 = l2
        self.washout = washout
        self.score_fn = score_fn
        self.mode = mode
        self.store_matrices = store_matrices

    def fit(self, model: Any, train: Any, val: Optional[Any] = None) -> FitResult:
        rng = get_rng_state()
        preprocess = lambda x: model.core(x).states  # noqa: E731
        device = next(model.parameters()).device

        if val is not None:
            if self.score_fn is None or self.mode is None:
                raise ValueError(
                    "Validation selection requires both score_fn and mode."
                )
            l2_values = self.l2 if isinstance(self.l2, list) else [self.l2]
            readout, best_l2, best_score, a, b = fit_and_validate_readout(
                train_loader=train,
                eval_loader=val,
                l2_values=l2_values,
                score_fn=lambda target, pred: self.score_fn(pred, target),
                mode=self.mode,
                preprocess_fn=preprocess,
                skip_first_n=self.washout,
                device=device,
            )
            best = {"l2": best_l2, "score": best_score}
        else:
            readout, a, b = fit_readout(
                train,
                preprocess_fn=preprocess,
                l2=self.l2,
                skip_first_n=self.washout,
                device=device,
            )
            best = {"l2": self.l2}

        model.head.set_weight(readout.T)
        extras = {"A": a, "B": b} if self.store_matrices else {}
        return FitResult(history={}, best=best, extras=extras, rng=rng)
