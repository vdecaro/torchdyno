"""Maps across the units at every timestep: mixing neighbours by a convolution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    import torch
    from torch import Tensor


class Mix(nn.Module):
    """Mix neighbouring units at every step, by one convolution across the units.

    The features hold ``channels`` blocks of units side by side, as a complex
    [`LinearDiagonalCell`][torchdyno.sequence.LinearDiagonalCell] stores
    ``[Re h; Im h]`` (``channels=2``). A kernel of ``kernel_size`` units slides
    across them with zero padding, the same at every step, and sums over the
    channels, mapping ``(T, B, channels · units)`` to ``(T, B, units)``. On
    ``[Re h; Im h]`` it is ``Re(w * h) + c`` for a complex kernel ``w``: the mixing
    layer of Pinna et al., "ParalESN: Enabling parallel information processing in
    Reservoir Computing" (2026), which follows it by ``nn.Tanh()``.

    A diagonal recurrence leaves its units uncoupled; this couples neighbours
    after every step instead of within it, so a [`Scan`][torchdyno.sequence.Scan]
    still runs the recurrence in parallel. The kernel trains unless frozen.
    """

    def __init__(
        self, channels: int = 1, kernel_size: int = 3, dtype: torch.dtype | None = None
    ):
        """Initializes the layer.

        Args:
            channels: the number of blocks of units in the features.
            kernel_size: the number of neighbouring units each output mixes.
            dtype: the data type; ``None`` takes torch's default.
        """
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv1d(channels, 1, kernel_size, padding="same", dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """Mix the units of ``x``.

        Args:
            x: the sequence, shape ``(T, B, channels · units)``.

        Returns:
            The mixed units, shape ``(T, B, units)``; NaN at the steps a sequence
            does not have.

        Raises:
            ValueError: if the features do not split into ``channels`` blocks.
        """
        steps, batch, features = x.shape
        if features % self.channels:
            raise ValueError(
                f"{features} features do not split into {self.channels} channels."
            )
        mixed = self.conv(x.reshape(steps * batch, self.channels, -1))
        return mixed.reshape(steps, batch, -1)
