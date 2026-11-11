"""OscillatorCell: coupled damped oscillators (coRNN)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torchdyno.matrices.ops import LinearMap
from torchdyno.sequence.cells.base import (
    Cell,
    _at,
    _keep,
    _width,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from torchdyno.matrices import LinearMap
    from torchdyno.sequence.cells.base import Step


class OscillatorCell(Cell):
    """Coupled damped oscillators, integrated with an explicit step (coRNN).

    From Rusch and Mishra, "Coupled Oscillatory Recurrent Neural Network
    (coRNN)" (2021). The state ``[y; z]`` holds positions and velocities::

        z' = z + dt (φ(W [y; z] + V u + b) - gamma y - ε z)
        y' = y + dt z'

    ``recurrent`` maps the whole state ``[y; z]`` to the positions, so
    ``W [y; z]`` is the paper's ``W y + W_z z`` as one product; ``input`` holds
    ``V`` and the bias ``b``. Each of ``dt``, ``gamma`` and ``epsilon`` is a
    constant, a per-unit tensor kept as a buffer, or a callable ``(u, state)``
    computed at every timestep.
    """

    def __init__(
        self,
        input: LinearMap,
        recurrent: LinearMap,
        dt: Step = 0.1,
        gamma: Step = 1.0,
        epsilon: Step = 1.0,
        activation: Callable[[Tensor], Tensor] = torch.tanh,
    ):
        """Initializes the cell.

        Args:
            input: the input matrix ``V``, with the bias ``b``.
            recurrent: the ``(2H -> H)`` matrix on the state ``[y; z]``.
            dt: the integration step.
            gamma: the restoring force.
            epsilon: the damping.
            activation: the activation ``φ``.

        Raises:
            ValueError: if ``dt``, ``gamma`` or ``epsilon`` is not positive, or a
                matrix does not match the state.
        """
        super().__init__()
        _width(
            "recurrent.in_features", recurrent.in_features, 2 * recurrent.out_features
        )
        _width("input.out_features", input.out_features, recurrent.out_features)
        self.input = input
        self.recurrent = recurrent
        _keep(self, "dt", dt)
        _keep(self, "gamma", gamma)
        _keep(self, "epsilon", epsilon)
        self.activation = activation
        self.positions = recurrent.out_features
        self.state_size = 2 * self.positions

    def forward(self, u: Tensor, state: Tensor) -> Tensor:
        """Advance the state by one timestep.

        Args:
            u: the input, shape ``(B, F)``.
            state: ``[y; z]``, shape ``(B, state_size)``.

        Returns:
            The new state, in the same layout.
        """
        y, z = state[..., : self.positions], state[..., self.positions :]
        dt, gamma, epsilon = (
            _at(value, u, state) for value in (self.dt, self.gamma, self.epsilon)
        )
        drive = self.activation(self.recurrent(state) + self.input(u))
        z = z + dt * (drive - gamma * y - epsilon * z)
        return torch.cat([y + dt * z, z], dim=-1)
