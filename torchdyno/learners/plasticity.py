"""IntrinsicPlasticity: unsupervised adaptation of cells' gains and biases."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
import torch.nn.functional as F

from torchdyno.learners.base import FitResult
from torchdyno.reproducibility import get_rng_state

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import (
        Tensor,
        nn,
    )


class IntrinsicPlasticity:
    """Push every tanh unit of a model toward a Gaussian output distribution.

    From Schrauwen et al., "Improving reservoirs using intrinsic plasticity"
    (2008). A cell takes part if it has a per-unit ``gain`` and ``bias`` and a
    ``preactivation(u, h)``, and its activation is tanh, as for
    :class:`~torchdyno.cells.LeakyCell`. Each unit's activation
    ``h = tanh(g x + c)``, with ``x`` the cell's preactivation, is moved toward
    ``N(mu, sigma²)`` by the closed-form gradients of the KL divergence::

        dc = -mu / sigma² + h / sigma² (2 sigma² + 1 - h² + mu h)
        dg = -1 / g + dc x

    applied online, unit by unit, after every timestep and averaged over the
    batch: ``c -= lr dc`` and ``g -= lr dg``. It is unsupervised (targets are
    ignored) and changes only the gains and biases, even if they are frozen to
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
        """Adapt every tanh cell of ``model`` on the inputs of ``train``.

        Args:
            model: the model; every cell with a ``gain``, a ``bias`` and a
                ``preactivation`` in it is adapted.
            train: batches of time-first ``(x, y)``; ``y`` is ignored.
            val: unused; accepted so every learner fits the same way.

        Returns:
            The RNG state at the start.

        Raises:
            ValueError: if the model has no such cell, or one is not tanh.
        """
        cells = [m for m in model.modules() if _adaptable(m)]
        if not cells:
            raise ValueError(
                "Intrinsic plasticity needs a cell with a gain, a bias and "
                "preactivation(u, h), such as LeakyCell."
            )
        if not all(_is_tanh(getattr(cell, "activation", None)) for cell in cells):
            raise ValueError("Intrinsic plasticity needs tanh activations.")
        rng = get_rng_state()
        handles = [cell.register_forward_hook(self._adapt) for cell in cells]
        try:
            with torch.no_grad():
                for _ in range(self.epochs):
                    for x, _ in train:
                        model(x)
        finally:
            for handle in handles:
                handle.remove()
        return FitResult(rng=rng)

    def _adapt(self, cell: Any, args: tuple[Tensor, Tensor], _: Tensor) -> None:
        """One online step on the units of ``cell``, after it ran a timestep."""
        x = cell.preactivation(*args)
        h = torch.tanh(cell.gain * x + cell.bias)
        var = self.sigma**2
        bias = -self.mu / var + (h / var) * (2 * var + 1 - h**2 + self.mu * h)
        gain = -1 / cell.gain + bias * x
        cell.bias.sub_(self.lr * bias.mean(0))
        cell.gain.sub_(self.lr * gain.mean(0))


def _adaptable(module: nn.Module) -> bool:
    """Whether ``module`` has the per-unit gain, bias and preactivation IP adapts."""
    return all(hasattr(module, name) for name in ("gain", "bias", "preactivation"))


def _is_tanh(activation: object) -> bool:
    """Whether ``activation`` is one of torch's tanh functions or modules."""
    return activation in (torch.tanh, F.tanh) or isinstance(activation, torch.nn.Tanh)
