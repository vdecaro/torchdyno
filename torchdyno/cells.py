"""Cells: one step of a recurrence, ``cell(u_t, h) -> h'``.

The new state is also the cell's output, so a layer such as
:class:`~torchdyno.layers.Recurrent` only has to collect the states. Cells take
their matrices as arguments, so the same update rule works with any matrix
structure from :mod:`torchdyno.matrices`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

    from torch import Tensor

    from torchdyno.matrices import LinearMap

    # a constant, a per-unit tensor, or computed from the input and the state
    Step: TypeAlias = float | Tensor | Callable[[Tensor, Tensor], Tensor]

__all__ = [
    "Cell",
    "ClosedFormCell",
    "EulerCell",
    "Gate",
    "LeakyCell",
    "LeakyField",
    "LinearDiagonalCell",
    "OscillatorCell",
    "RingDiagonal",
    "ZOHDiagonal",
]


class Cell(nn.Module):
    """Base class for cells: ``forward(u_t, h) -> h'`` on ``(B, state_size)`` states."""

    state_size: int


def _width(name: str, actual: int, expected: int) -> None:
    """Raise unless a part of a cell has the width the state needs."""
    if actual != expected:
        raise ValueError(f"{name} has width {actual}, but the state needs {expected}.")


def _keep(module: nn.Module, name: str, value: Step) -> None:
    """Keep a step-like ``value`` on ``module``: a tensor as a buffer.

    A constant or a tensor must be positive; a callable ``value(u, h)`` is
    computed at every timestep.
    """
    if not callable(value) and not torch.all(torch.as_tensor(value) > 0):
        raise ValueError(f"{name} must be > 0, got {value}.")
    if isinstance(value, torch.Tensor):
        module.register_buffer(name, value)
    else:
        setattr(module, name, value)


def _at(value: Step, u: Tensor, h: Tensor) -> float | Tensor:
    """A step-like value at the current timestep."""
    return value(u, h) if callable(value) else value


class Gate(nn.Module):
    """A per-unit step size in ``(lo, hi)``, driven by the input and the state.

    ``lo + (hi - lo) * sigmoid(W u + b + a ⊙ h)``, with ``a`` starting at zero so
    the step size starts input-only.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bounds: tuple[float, float] = (1e-5, 0.2),
        dtype: torch.dtype | None = None,
    ):
        """Initializes the gate.

        Args:
            in_features: the input width.
            out_features: the state width.
            bounds: the ``(lo, hi)`` range of the step size.
            dtype: the data type; ``None`` takes torch's default.

        Raises:
            ValueError: unless ``0 <= lo < hi``.
        """
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        if not 0 <= bounds[0] < bounds[1]:
            raise ValueError(f"Need 0 <= lo < hi, got {bounds}.")
        self.in_features, self.out_features = in_features, out_features
        self.lo, self.hi = bounds
        self.input = nn.Linear(in_features, out_features, dtype=dtype)
        self.state = nn.Parameter(torch.zeros(out_features, dtype=dtype))

    def forward(self, u: Tensor, h: Tensor) -> Tensor:
        """Compute the step size for one timestep.

        Args:
            u: the input, shape ``(B, in_features)``.
            h: the state, shape ``(B, out_features)``.

        Returns:
            The step size, shape ``(B, out_features)``.
        """
        logits = self.input(u) + self.state * h
        return self.lo + (self.hi - self.lo) * torch.sigmoid(logits)


class LeakyField(nn.Module):
    """A leaky ODE's vector field: ``f(u, v) = -v + W φ(v) + L v + B u``.

    With a block-diagonal ``W`` and a skew-symmetric ``L``, an assembly of
    coupled modules; integrate it with :class:`EulerCell`.
    """

    def __init__(
        self,
        input: LinearMap,
        recurrent: LinearMap,
        coupling: LinearMap | None = None,
        activation: Callable[[Tensor], Tensor] = torch.relu,
    ):
        """Initializes the field.

        Args:
            input: the input matrix ``B``.
            recurrent: the matrix ``W`` applied to ``φ(v)``.
            coupling: an optional matrix ``L`` applied to ``v``.
            activation: the activation ``φ``.

        Raises:
            ValueError: if a matrix does not match the state.
        """
        super().__init__()
        size = recurrent.out_features
        _width("recurrent.in_features", recurrent.in_features, size)
        _width("input.out_features", input.out_features, size)
        if coupling is not None:
            _width("coupling.in_features", coupling.in_features, size)
            _width("coupling.out_features", coupling.out_features, size)
        self.input = input
        self.recurrent = recurrent
        self.coupling = coupling
        self.activation = activation
        self.state_size = size

    def forward(self, u: Tensor, v: Tensor) -> Tensor:
        """The time derivative of the state.

        Args:
            u: the input, shape ``(B, F)``.
            v: the state, shape ``(B, state_size)``.

        Returns:
            ``f(u, v)``, shape ``(B, state_size)``.
        """
        drive = -v + self.recurrent(self.activation(v))
        if self.coupling is not None:
            drive = drive + self.coupling(v)
        return drive + self.input(u)


class EulerCell(Cell):
    """One explicit Euler step of an ODE: ``h' = h + ε ⊙ f(u, h)``.

    ``f`` is any vector field, e.g. :class:`LeakyField` for an assembly of
    coupled modules, or a module of one's own. ``ε`` is a constant, a per-unit
    tensor, or computed from the input and the state, e.g. by a :class:`Gate`.
    """

    step: Step

    def __init__(
        self,
        field: Callable[[Tensor, Tensor], Tensor],
        step: Step = 0.01,
        state_size: int | None = None,
    ):
        """Initializes the cell.

        Args:
            field: the vector field ``f(u, h)``.
            step: the step size ``ε``: a constant, a tensor kept as a buffer, or
                a callable ``step(u, h) -> ε``.
            state_size: the state width; ``None`` takes ``field.state_size``.

        Raises:
            ValueError: if the state width is unknown, the step does not match
                it, or a constant step is not positive.
        """
        super().__init__()
        size = getattr(field, "state_size", None) if state_size is None else state_size
        if not isinstance(size, int):
            raise ValueError("Give state_size= for a field without a state_size.")
        step_width = getattr(step, "out_features", None)
        if isinstance(step_width, int):
            _width("step.out_features", step_width, size)
        if isinstance(step, torch.Tensor) and step.numel() != 1:
            _width("step", step.shape[-1], size)
        self.field = field
        _keep(self, "step", step)
        self.state_size = size

    def forward(self, u: Tensor, h: Tensor) -> Tensor:
        """Advance the state by one timestep.

        Args:
            u: the input, shape ``(B, F)``.
            h: the state, shape ``(B, state_size)``.

        Returns:
            The new state.
        """
        return h + _at(self.step, u, h) * self.field(u, h)


class LeakyCell(Cell):
    """A leaky-integrator unit: ``h' = (1 - a) h + a φ(g ⊙ (B u + W h) + c)``.

    The gain ``g`` and bias ``c`` start at one and zero, an exact no-op, and
    change only under intrinsic plasticity or backprop.
    """

    leak: float | Tensor

    def __init__(
        self,
        input: LinearMap,
        recurrent: LinearMap,
        leak: float | Tensor = 1.0,
        activation: Callable[[Tensor], Tensor] = torch.tanh,
    ):
        """Initializes the cell.

        Args:
            input: the input matrix ``B``, with a bias if the units need one.
            recurrent: the recurrent matrix ``W``.
            leak: the leak rate ``a``: a constant, or a per-unit tensor kept as
                a buffer.
            activation: the activation ``φ``.

        Raises:
            ValueError: if a matrix or the leak does not match the state, or the
                leak is outside ``(0, 1]``.
        """
        super().__init__()
        rate = torch.as_tensor(leak)
        if not torch.all((rate > 0) & (rate <= 1)):
            raise ValueError(f"leak must be in (0, 1], got {leak}.")
        size = recurrent.out_features
        _width("recurrent.in_features", recurrent.in_features, size)
        _width("input.out_features", input.out_features, size)
        if isinstance(leak, torch.Tensor) and leak.numel() != 1:
            _width("leak", leak.shape[-1], size)
        self.input = input
        self.recurrent = recurrent
        if isinstance(leak, torch.Tensor):
            self.register_buffer("leak", leak)
        else:
            self.leak = leak
        self.activation = activation
        self.state_size = recurrent.out_features
        weight = next(recurrent.parameters(), None)
        dtype = torch.get_default_dtype() if weight is None else weight.dtype
        self.gain = nn.Parameter(torch.ones(self.state_size, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(self.state_size, dtype=dtype))

    def preactivation(self, u: Tensor, h: Tensor) -> Tensor:
        """The input to the gain: ``B u + W h``.

        Args:
            u: the input, shape ``(B, F)``.
            h: the state, shape ``(B, state_size)``.

        Returns:
            The preactivation, shape ``(B, state_size)``.
        """
        return self.input(u) + self.recurrent(h)

    def forward(self, u: Tensor, h: Tensor) -> Tensor:
        """Advance the state by one timestep.

        Args:
            u: the input, shape ``(B, F)``.
            h: the state, shape ``(B, state_size)``.

        Returns:
            The new state.
        """
        drive = self.activation(self.gain * self.preactivation(u, h) + self.bias)
        return (1 - self.leak) * h + self.leak * drive


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


class RingDiagonal(nn.Module):
    """LRU's stable complex diagonal ``λ = exp(-e^nu + iθ)``, with ``sqrt(1 - |λ|²)``.

    From Orvieto et al., "Resurrecting Recurrent Neural Networks for Long
    Sequences" (2023). ``|λ| < 1`` for every ``nu``, so training cannot make the
    recurrence unstable. ``|λ|`` is drawn uniformly on the ring
    ``[r_min, r_max]`` and the phase on ``[0, max_phase]``.
    """

    def __init__(
        self,
        size: int,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 2 * math.pi,
        dtype: torch.dtype | None = None,
    ):
        """Initializes the transition.

        Args:
            size: the number of modes.
            r_min: the smallest initial magnitude.
            r_max: the largest initial magnitude.
            max_phase: the largest initial phase, in radians.
            dtype: the real data type of ``nu`` and ``θ``; ``None`` takes
                torch's default.

        Raises:
            ValueError: unless ``0 <= r_min <= r_max <= 1`` and ``r_min < 1``.
        """
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        if not (0.0 <= r_min <= r_max <= 1.0 and r_min < 1.0):
            raise ValueError(
                f"Need 0 <= r_min <= r_max <= 1 with r_min < 1, got {r_min}, {r_max}."
            )
        # in float64, so a draw just below r_max = 1 keeps a finite nu
        draw = torch.rand(size, dtype=dtype).to(torch.float64)
        magnitude_sq = draw * (r_max**2 - r_min**2) + r_min**2
        self.nu = nn.Parameter(torch.log(-0.5 * torch.log(magnitude_sq)).to(dtype))
        self.theta = nn.Parameter(max_phase * torch.rand(size, dtype=dtype))

    def forward(self, u: Tensor | None = None) -> tuple[Tensor, Tensor]:  # noqa: ARG002
        """The multiplier and the input scale.

        Args:
            u: the input; unused, as the transition does not depend on it.

        Returns:
            ``(λ, sqrt(1 - |λ|²))``, each of shape ``(size,)``.
        """
        lam = torch.polar(torch.exp(-torch.exp(self.nu)), self.theta)
        return lam, torch.sqrt((1 - lam.abs() ** 2).clamp_min(0))


class ZOHDiagonal(nn.Module):
    """S4D's diagonal ``A = -e^a + ib``, discretised by zero-order hold with step ``Δ``.

    From Gu et al., "On the Parameterization and Initialization of Diagonal
    State Space Models" (2022), with the S4D-Lin initialisation
    ``A_n = -1/2 + iπn``. The step ``Δ = e^d`` is learnt per mode, drawn
    log-uniformly in ``[dt_min, dt_max]``.
    """

    def __init__(
        self,
        size: int,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dtype: torch.dtype | None = None,
    ):
        """Initializes the transition.

        Args:
            size: the number of modes.
            dt_min: the smallest initial step.
            dt_max: the largest initial step.
            dtype: the real data type of the parameters; ``None`` takes
                torch's default.

        Raises:
            ValueError: unless ``0 < dt_min <= dt_max``.
        """
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        if not 0.0 < dt_min <= dt_max:
            raise ValueError(f"Need 0 < dt_min <= dt_max, got {dt_min}, {dt_max}.")
        self.log_real = nn.Parameter(torch.full((size,), math.log(0.5), dtype=dtype))
        self.imag = nn.Parameter(math.pi * torch.arange(size, dtype=dtype))
        low, high = math.log(dt_min), math.log(dt_max)
        self.log_dt = nn.Parameter(low + torch.rand(size, dtype=dtype) * (high - low))

    def forward(self, u: Tensor | None = None) -> tuple[Tensor, Tensor]:  # noqa: ARG002
        """The multiplier and the input scale.

        Args:
            u: the input; unused, as the transition does not depend on it.

        Returns:
            ``(Ā, (Ā - 1) / A)`` with ``Ā = exp(ΔA)``, each of shape ``(size,)``.
        """
        a = torch.complex(-torch.exp(self.log_real), self.imag)
        dt = torch.exp(self.log_dt)
        # (Ā - 1) / A by expm1, in complex128: small ΔA loses exp - 1 to cancellation
        a128 = a.to(torch.complex128)
        scale = torch.expm1(dt.to(torch.float64) * a128) / a128
        return torch.exp(dt * a), scale.to(a.dtype)


class LinearDiagonalCell(Cell):
    """A linear recurrence on a diagonal: ``h' = a ⊙ h + s ⊙ (B u)``.

    ``transition(u)`` gives ``(a, s)``, broadcastable to ``u.shape[:-1] +
    (modes,)``: constant, as :class:`RingDiagonal` (LRU) and
    :class:`ZOHDiagonal` (S4D) are, or computed from the input, as a selective
    step is. ``input`` is ``B``. If ``a``, ``s`` or ``B u`` is complex, the
    state is stored as the real ``[Re h; Im h]`` and ``state_size`` is twice
    the number of modes; otherwise it is ``h`` itself. Because the recurrence
    is linear, :class:`~torchdyno.layers.Scan` can run it in parallel over
    time.
    """

    def __init__(self, transition: nn.Module, input: LinearMap):
        """Initializes the cell.

        Args:
            transition: a module whose ``forward(u)`` returns ``(a, s)``.
            input: the input matrix ``B``.

        Raises:
            ValueError: if the transition has another number of modes than the
                input matrix.
        """
        super().__init__()
        self.transition = transition
        self.input = input
        self.modes = input.out_features
        weight = next(input.parameters(), None)
        dtype = torch.get_default_dtype() if weight is None else weight.dtype
        with torch.no_grad():
            probe = torch.zeros(1, input.in_features, dtype=dtype)
            a, s = transition(probe)
            _width("transition", a.shape[-1], self.modes)
            parts = (a, s, input(probe))
        self.is_complex = any(part.is_complex() for part in parts)
        self.state_size = 2 * self.modes if self.is_complex else self.modes

    def coefficients(self, u: Tensor) -> tuple[Tensor, Tensor]:
        """The multiplier and the addend of the recurrence for input ``u``.

        Args:
            u: the input, with ``F`` as the last dimension.

        Returns:
            ``(a, s ⊙ B u)``: ``a`` broadcastable to the addend, which has shape
            ``u.shape[:-1] + (modes,)``.
        """
        a, s = self.transition(u)
        return a, s * self.input(u)

    def unpack(self, state: Tensor) -> Tensor:
        """The state ``h`` from its stored form."""
        if not self.is_complex:
            return state
        return torch.complex(state[..., : self.modes], state[..., self.modes :])

    def pack(self, h: Tensor) -> Tensor:
        """The stored form of the state ``h``."""
        return torch.cat([h.real, h.imag], dim=-1) if self.is_complex else h

    def forward(self, u: Tensor, state: Tensor) -> Tensor:
        """Advance the state by one timestep.

        Args:
            u: the input, shape ``(B, F)``.
            state: the stored state, shape ``(B, state_size)``.

        Returns:
            The new state, in the same layout.
        """
        a, b = self.coefficients(u)
        return self.pack(a * self.unpack(state) + b)
