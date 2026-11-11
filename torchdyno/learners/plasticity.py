"""IntrinsicPlasticity: unsupervised adaptation of leaky cells' gain and bias."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
import torch.nn.functional as F

from torchdyno.cells import LeakyCell
from torchdyno.learners.base import FitResult
from torchdyno.reproducibility import get_rng_state

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import (
        Tensor,
        nn,
    )


class IntrinsicPlasticity:
    """Push every tanh :class:`~torchdyno.cells.LeakyCell` toward a Gaussian output.

    From Schrauwen et al., "Improving reservoirs using intrinsic plasticity"
    (2008). Each unit's activation ``h = tanh(g x + c)``, with ``x`` the
    cell's preactivation, is moved toward ``N(mu, sigma²)`` by the closed-form
    gradients of the KL divergence::

        dc = -mu / sigma² + h / sigma² (2 sigma² + 1 - h² + mu h)
        dg = -1 / g + dc x

    summed over time, averaged over the batch, normalised, and applied as an
    SGD step after every batch. It is unsupervised (targets are ignored) and
    changes only the gain ``g`` and bias ``c``, even if they are frozen to
    backprop. Fit the readout afterwards, e.g. with
    :class:`~torchdyno.learners.Ridge`.
    """

    def __init__(
        self, lr: float = 1e-3, mu: float = 0.0, sigma: float = 0.2, epochs: int = 1
    ):
        """Initializes the learner.

        Args:
            lr: the step size.
            mu: the target mean of the activations.
            sigma: the target standard deviation of the activations.
            epochs: passes over ``train``.

        Raises:
            ValueError: if ``sigma`` is not positive.
        """
        if not sigma > 0:
            raise ValueError(f"sigma must be > 0, got {sigma}.")
        self.lr = lr
        self.mu = mu
        self.sigma = sigma
        self.epochs = epochs

    def fit(
        self,
        model: nn.Module,
        train: Iterable[tuple[Tensor, Tensor]],
        val: Iterable[tuple[Tensor, Tensor]] | None = None,  # noqa: ARG002
    ) -> FitResult:
        """Adapt every tanh leaky cell of ``model`` on the inputs of ``train``.

        Args:
            model: the model; every :class:`~torchdyno.cells.LeakyCell` in it is
                adapted.
            train: batches of time-first ``(x, y)``; ``y`` is ignored.
            val: unused; accepted so every learner fits the same way.

        Returns:
            The RNG state at the start.

        Raises:
            ValueError: if the model has no leaky cell, or one is not tanh.
        """
        cells = [m for m in model.modules() if isinstance(m, LeakyCell)]
        if not cells:
            raise ValueError("Intrinsic plasticity needs a LeakyCell in the model.")
        if not all(_is_tanh(cell.activation) for cell in cells):
            raise ValueError("Intrinsic plasticity needs tanh activations.")
        rng = get_rng_state()
        traces: dict[LeakyCell, list[tuple[Tensor, Tensor]]] = {c: [] for c in cells}

        def record(cell: Any, args: tuple[Tensor, Tensor], _: Tensor) -> None:
            x = cell.preactivation(*args)
            traces[cell].append((x, torch.tanh(cell.gain * x + cell.bias)))

        handles = [cell.register_forward_hook(record) for cell in cells]
        try:
            with torch.no_grad():
                for _ in range(self.epochs):
                    for x, _ in train:
                        model(x)
                        for cell, trace in traces.items():
                            self._update(cell, trace)
                            trace.clear()
        finally:
            for handle in handles:
                handle.remove()
        return FitResult(rng=rng)

    def _update(self, cell: LeakyCell, trace: list[tuple[Tensor, Tensor]]) -> None:
        x = torch.stack([pre for pre, _ in trace])
        h = torch.stack([out for _, out in trace])
        var = self.sigma**2
        bias = -self.mu / var + (h / var) * (2 * var + 1 - h**2 + self.mu * h)
        gain = -1 / cell.gain + bias * x
        cell.bias.sub_(self.lr * F.normalize(bias.sum(0).mean(0), dim=0))
        cell.gain.sub_(self.lr * F.normalize(gain.sum(0).mean(0), dim=0))


def _is_tanh(activation: object) -> bool:
    """Whether ``activation`` is one of torch's tanh functions or modules."""
    return activation in (torch.tanh, F.tanh) or isinstance(activation, torch.nn.Tanh)
