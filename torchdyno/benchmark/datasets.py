"""Synthetic benchmarks: chaotic systems, NARMA and memory capacity, split in time."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torchdyno.benchmark.dataset import BenchmarkDataset
from torchdyno.objectives import (
    MSE,
    NRMSE,
    MemoryCapacity,
    Regression,
    Task,
)


def lorenz_system(
    length: int,
    dt: float = 0.01,
    sigma: float = 10.0,
    beta: float = 8 / 3,
    rho: float = 28.0,
    start: tuple[float, float, float] = (0.1, 0.1, 0.1),
) -> torch.Tensor:
    """Integrate the Lorenz system with forward Euler.

    ``dx/dt = sigma (y - x)``, ``dy/dt = x (rho - z) - y``,
    ``dz/dt = x y - beta z``.

    Args:
        length: number of steps.
        dt: the step size.
        sigma: the sigma parameter.
        beta: the beta parameter.
        rho: the rho parameter.
        start: the initial ``(x, y, z)``.

    Returns:
        The ``(length, 3)`` trajectory, divided by its largest absolute value.
    """
    x, y, z = start
    points = []
    for _ in range(length):
        x, y, z = (
            x + dt * sigma * (y - x),
            y + dt * (x * (rho - z) - y),
            z + dt * (x * y - beta * z),
        )
        points.append((x, y, z))
    trajectory = torch.tensor(points)
    return trajectory / trajectory.abs().max()


def mackey_glass(
    length: int,
    tau: float = 17.0,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: int = 10,
    step: float = 0.1,
    sample: float = 1.0,
    start: float = 1.2,
) -> torch.Tensor:
    """Integrate the Mackey-Glass delay equation with forward Euler.

    ``dx/dt = beta x(t - tau) / (1 + x(t - tau)^n) - gamma x(t)``, from
    ``x = start`` at and before time zero; ``tau = 17`` is mildly chaotic.

    Args:
        length: number of samples.
        tau: the delay.
        beta: the production rate.
        gamma: the decay rate.
        n: the nonlinearity's exponent.
        step: the integration step; ``tau`` and ``sample`` should be multiples.
        sample: the time between samples.
        start: the constant history before time zero.

    Returns:
        The ``(length,)`` series.
    """
    lag, every = round(tau / step), round(sample / step)
    history = [start] * (lag + 1)
    samples = []
    for i in range(length * every):
        x, delayed = history[-1], history[-1 - lag]
        history.append(x + step * (beta * delayed / (1 + delayed**n) - gamma * x))
        if (i + 1) % every == 0:
            samples.append(history[-1])
        del history[0]
    return torch.tensor(samples)


def narma(inputs: torch.Tensor, order: int = 10) -> torch.Tensor:
    """The NARMA system of a given order, driven by ``inputs``.

    ``y(t + 1) = 0.3 y(t) + 0.05 y(t) Σ_{i < order} y(t - i) + 1.5 u(t - order + 1)
    u(t) + 0.1``, from ``y = 0``; with ``u ~ U(0, 0.5)`` and order 10 it is the
    usual NARMA10. Higher orders can diverge.

    Args:
        inputs: the ``(T,)`` input ``u``.
        order: how many past outputs and inputs the next output depends on.

    Returns:
        The ``(T,)`` output: ``y(t + 1)``, the target at step ``t``.
    """
    u = inputs.tolist()
    y = [0.0] * (len(u) + 1)  # y[t] for t = 0, ..., T
    for t in range(order - 1, len(u)):
        recent = sum(y[t - order + 1 : t + 1])
        y[t + 1] = (
            0.3 * y[t] + 0.05 * y[t] * recent + 1.5 * u[t - order + 1] * u[t] + 0.1
        )
    return torch.tensor(y[1:], dtype=inputs.dtype)


def _split_in_time(
    name: str,
    task: Task,
    x: torch.Tensor,
    y: torch.Tensor,
    splits: tuple[float, float],
    washout: int,
) -> BenchmarkDataset:
    """Contiguous train, validation and evaluation segments, each one batch."""
    length = len(x)
    n_train, n_val = int(length * splits[0]), int(length * splits[1])
    cuts = [(0, n_train), (n_train, n_train + n_val), (n_train + n_val, length)]
    train, val, test = ([(x[a:b], y[a:b])] for a, b in cuts)
    return BenchmarkDataset(
        name=name,
        task=task,
        train_loader=train,
        eval_loader=test,
        val_loader=val,
        washout=washout,
    )


def lorenz_benchmark(
    length: int = 3000,
    target_delay: int = 1,
    splits: tuple[float, float] = (0.5, 0.25),
    washout: int = 0,
) -> BenchmarkDataset:
    """Lorenz forecasting: predict the state ``target_delay`` steps ahead.

    The trajectory is cut into contiguous train, validation and evaluation
    segments, each a single ``(T, 1, 3)`` batch.

    Args:
        length: number of integration steps.
        target_delay: how many steps ahead the target is.
        splits: the train and validation fractions; evaluation gets the rest.
        washout: leading timesteps of each segment left unscored.

    Returns:
        The benchmark dataset.
    """
    trajectory = lorenz_system(length + target_delay).unsqueeze(1)
    x, y = trajectory[:-target_delay], trajectory[target_delay:]
    return _split_in_time("lorenz", Regression(), x, y, splits, washout)


def mackey_glass_benchmark(
    length: int = 3000,
    target_delay: int = 1,
    splits: tuple[float, float] = (0.5, 0.25),
    washout: int = 0,
    tau: float = 17.0,
) -> BenchmarkDataset:
    """Mackey-Glass forecasting: predict the series ``target_delay`` steps ahead.

    Split as [`lorenz_benchmark()`][torchdyno.benchmark.lorenz_benchmark], in ``(T, 1,
    1)`` batches.

    Args:
        length: number of samples.
        target_delay: how many steps ahead the target is.
        splits: the train and validation fractions; evaluation gets the rest.
        washout: leading timesteps of each segment left unscored.
        tau: the delay of the equation.

    Returns:
        The benchmark dataset.
    """
    series = mackey_glass(length + target_delay, tau=tau)[:, None, None]
    x, y = series[:-target_delay], series[target_delay:]
    return _split_in_time("mackey_glass", Regression(), x, y, splits, washout)


def narma_benchmark(
    length: int = 3000,
    order: int = 10,
    splits: tuple[float, float] = (0.5, 0.25),
    washout: int = 0,
) -> BenchmarkDataset:
    """NARMA: from inputs ``u ~ U(0, 0.5)``, predict the system's next output.

    The inputs are drawn from torch's generator; split as
    [`lorenz_benchmark()`][torchdyno.benchmark.lorenz_benchmark], in ``(T, 1, 1)``
    batches.

    Args:
        length: number of steps.
        order: the order of the system, see [`narma()`][torchdyno.benchmark.narma].
        splits: the train and validation fractions; evaluation gets the rest.
        washout: leading timesteps of each segment left unscored.

    Returns:
        The benchmark dataset.
    """
    u = torch.rand(length) * 0.5
    x, y = u[:, None, None], narma(u, order)[:, None, None]
    return _split_in_time(f"narma{order}", Regression(), x, y, splits, washout)


def memory_capacity_benchmark(
    length: int = 3000,
    delays: int = 50,
    splits: tuple[float, float] = (0.5, 0.25),
    washout: int | None = None,
) -> BenchmarkDataset:
    """Memory capacity: recall the input of ``1, ..., delays`` steps ago.

    Jaeger's measure of short-term memory. The inputs ``u ~ U(-0.8, 0.8)`` are drawn
    from torch's generator, and output ``k`` is trained on ``u(t - k)``, so the primary
    metric, [`MemoryCapacity`][torchdyno.objectives.MemoryCapacity], is at most
    ``delays``. Split as [`lorenz_benchmark()`][torchdyno.benchmark.lorenz_benchmark],
    in ``(T, 1, ·)`` batches.

    Args:
        length: number of steps.
        delays: the number of delays, one output each.
        splits: the train and validation fractions; evaluation gets the rest.
        washout: leading timesteps of each segment left unscored; ``None``
            takes ``delays``, so every scored target is a real past input.

    Returns:
        The benchmark dataset.
    """
    u = torch.rand(length) * 1.6 - 0.8
    y = torch.stack([F.pad(u, (k, 0))[:length] for k in range(1, delays + 1)], -1)
    task = Task(
        {"memory_capacity": MemoryCapacity(), "nrmse": NRMSE(), "mse": MSE()},
        "memory_capacity",
        loss=F.mse_loss,
    )
    washout = delays if washout is None else washout
    return _split_in_time(
        "memory_capacity", task, u[:, None, None], y[:, None], splits, washout
    )
