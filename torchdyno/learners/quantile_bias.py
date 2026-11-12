"""QuantileBias: set convolution biases from quantiles of their outputs."""

from __future__ import annotations

import math
from contextlib import suppress
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from torchdyno.learners.protocol import FitResult
from torchdyno.learners.scoring import restore_modes
from torchdyno.reproducibility import get_rng_state
from torchdyno.sequence._checks import _absent
from torchdyno.sequence.convolution import (
    RandomConv,
    _unproduced,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import (
        Tensor,
        nn,
    )


class _Captured(Exception):
    """Stops the model once the convolution's input is captured."""


class QuantileBias:
    """Set each kernel's bias so it is exceeded by a chosen share of its outputs.

    MiniRocket's bias rule (Dempster, Schmidt and Webb, 2021): kernel ``k`` is
    run, with padding, on one random training sequence, and its bias set to
    minus the ``q_k`` quantile of the outputs, so that [`PPV`][torchdyno.readout.PPV]
    of the kernel is about ``1 - q_k`` there. ``q_k`` is the fractional part of
    ``(k + 1) φ``, with ``φ`` the golden ratio, and kernels with the same
    weights and dilation share the sequence. Pair it with
    [`minirocket()`][torchdyno.sequence.minirocket] and chain it before the readout's
    learner.
    """

    def __init__(self, conv: str | None = None):
        """Initializes the learner.

        Args:
            conv: the convolution's name in the model, as
                ``model.get_submodule`` takes it; ``None`` takes the only
                [`RandomConv`][torchdyno.sequence.RandomConv].
        """
        self.conv = conv

    def _find(self, model: nn.Module) -> RandomConv:
        if self.conv is not None:
            found = [model.get_submodule(self.conv)]
        else:
            found = [m for m in model.modules() if isinstance(m, RandomConv)]
        if len(found) != 1 or not isinstance(found[0], RandomConv):
            raise TypeError(
                f"Expected one RandomConv, found {len(found)}; name it with conv=."
            )
        return found[0]

    def _inputs(self, model: nn.Module, conv: RandomConv, train: Iterable) -> Tensor:
        """Every training sequence as it reaches ``conv``, padded with NaN."""

        def capture(_conv: nn.Module, args: tuple[Tensor]) -> None:
            inputs.append(args[0].detach())
            raise _Captured

        inputs: list[Tensor] = []
        handle = conv.register_forward_pre_hook(capture)
        try:
            with restore_modes(model), torch.no_grad():
                for x, _ in train:
                    with suppress(_Captured):
                        model.eval()(x)
        finally:
            handle.remove()
        if not inputs:
            raise ValueError("train has no samples to fit on.")
        steps = max(len(x) for x in inputs)
        return torch.cat(
            [
                F.pad(x, (0, 0, 0, 0, 0, steps - len(x)), value=torch.nan)
                for x in inputs
            ],
            dim=1,
        )

    @torch.no_grad()
    def fit(
        self,
        model: nn.Module,
        train: Iterable[tuple[Tensor, Tensor]],
        val: Iterable[tuple[Tensor, Tensor]] | None = None,  # noqa: ARG002
    ) -> FitResult:
        """Set the biases of the model's convolution from ``train``.

        Args:
            model: the model holding one [`RandomConv`][torchdyno.sequence.RandomConv].
            train: batches of time-first ``(x, y)``; only ``x`` is used.
            val: unused; accepted so every learner fits the same way.

        Returns:
            The RNG state at the start.

        Raises:
            TypeError: if the model has no single random convolution.
            ValueError: if ``train`` has no samples.
        """
        conv = self._find(model)
        rng = get_rng_state()
        x = self._inputs(model, conv, train)
        keys = torch.cat([conv.weight.flatten(1), conv.dilations[:, None]], dim=1)
        _, pairs = keys.unique(dim=0, return_inverse=True)
        chosen = torch.randint(x.shape[1], (int(pairs.max()) + 1,))[pairs]
        golden = (1 + math.sqrt(5)) / 2
        quantiles = (
            torch.arange(1, conv.kernels + 1, dtype=torch.float64) * golden
        ) % 1
        groups = torch.stack([conv.lengths, conv.dilations], dim=1)
        for group in groups.unique(dim=0):
            index = (groups == group).all(dim=1).nonzero().squeeze(1)
            size, dilation = group.tolist()
            padding = (size - 1) * dilation // 2
            signal = x[:, chosen[index]]  # (T, n, C): each kernel's sequence
            absent = _absent(signal).T[:, None]  # (n, 1, T)
            signal = signal.permute(1, 2, 0).masked_fill(absent, 0)
            out = F.conv1d(  # one group per kernel, on its own sequence
                signal.reshape(1, -1, signal.shape[-1]),
                conv.weight[index, :, :size],
                padding=padding,
                dilation=dilation,
                groups=len(index),
            )[0]
            out = out.masked_fill(
                _unproduced(absent, size, dilation, padding)[:, 0], torch.nan
            )
            for k, row in zip(index.tolist(), out.double(), strict=True):
                conv.bias[k] = -torch.nanquantile(row, quantiles[k])
        return FitResult(rng=rng)
