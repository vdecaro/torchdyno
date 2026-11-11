"""ClosedFormCell: a closed-form continuous-time cell (CfC)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torchdyno.matrices.ops import LinearMap
from torchdyno.sequence.cells.base import (
    Cell,
    _at,
    _keep,
)

if TYPE_CHECKING:
    from torch import (
        Tensor,
        nn,
    )

    from torchdyno.matrices import LinearMap
    from torchdyno.sequence.cells.base import Step


class ClosedFormCell(Cell):
    """A closed-form continuous-time cell (CfC).

    From Hasani et al., "Closed-form continuous-time neural networks" (2022).
    A backbone reads ``[h; u]``, and four heads stacked in ``heads`` give
    ``f_1, f_2, t_a, t_b``::

        g  = sigmoid(t_a dt + t_b)
        h' = tanh(f_1) (1 - g) + tanh(f_2) g
    """

    def __init__(self, backbone: nn.Module, heads: LinearMap, dt: Step = 1.0):
        """Initializes the cell.

        Args:
            backbone: a module on ``[h; u]``, e.g. ``nn.Sequential(nn.Linear,
                nn.Tanh)``, or ``nn.Identity()``.
            heads: the map from the backbone to ``[f_1; f_2; t_a; t_b]``; the
                state size is a quarter of its output.
            dt: the elapsed time per step: a constant, a per-unit tensor, or a
                callable ``dt(u, h)``, e.g. reading each sample's elapsed time
                from an input channel for irregular sampling.

        Raises:
            ValueError: if ``dt`` is not positive, or ``heads`` does not output
                four equal parts.
        """
        super().__init__()
        if heads.out_features % 4:
            raise ValueError(
                f"heads must output four equal parts, got {heads.out_features}."
            )
        self.backbone = backbone
        self.heads = heads
        _keep(self, "dt", dt)
        self.state_size = heads.out_features // 4

    def forward(self, u: Tensor, h: Tensor) -> Tensor:
        """Advance the state by one timestep.

        Args:
            u: the input, shape ``(B, F)``.
            h: the state, shape ``(B, state_size)``.

        Returns:
            The new state.
        """
        f1, f2, t_a, t_b = self.heads(self.backbone(torch.cat([h, u], dim=-1))).chunk(
            4, -1
        )
        gate = torch.sigmoid(t_a * _at(self.dt, u, h) + t_b)
        return torch.tanh(f1) * (1 - gate) + torch.tanh(f2) * gate
