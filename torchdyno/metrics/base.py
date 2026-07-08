"""The Metric base: a named scalar score with a selection direction."""

from typing import Literal

from torch import Tensor


class Metric:
    """A named scalar score, callable as ``metric(pred, target) -> float``.

    ``mode`` says whether a smaller (``"min"``) or larger (``"max"``) value is
    better, so a learner can select the best hyperparameter on a val set. A
    Metric is a plain callable and drops straight into a learner's ``score_fn``.
    """

    name: str = ""
    mode: Literal["min", "max"] = "min"

    def __call__(self, pred: Tensor, target: Tensor) -> float:
        raise NotImplementedError
