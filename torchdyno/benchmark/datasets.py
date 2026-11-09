"""Dataset providers: build a BenchmarkDataset with fixed splits."""

from __future__ import annotations

import torch

from torchdyno.benchmark.dataset import BenchmarkDataset
from torchdyno.tasks import Regression


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


def lorenz_benchmark(
    length: int = 3000,
    target_delay: int = 1,
    splits: tuple[float, float] = (0.5, 0.25),
) -> BenchmarkDataset:
    """Lorenz forecasting: predict the state ``target_delay`` steps ahead.

    The trajectory is cut into contiguous train, validation and evaluation
    segments, each a single ``(T, 1, 3)`` batch.

    Args:
        length: number of integration steps.
        target_delay: how many steps ahead the target is.
        splits: the train and validation fractions; evaluation gets the rest.

    Returns:
        The benchmark dataset.
    """
    trajectory = lorenz_system(length + target_delay).unsqueeze(1)
    x, y = trajectory[:-target_delay], trajectory[target_delay:]
    n_train, n_val = int(length * splits[0]), int(length * splits[1])
    cuts = [(0, n_train), (n_train, n_train + n_val), (n_train + n_val, length)]
    train, val, test = ([(x[a:b], y[a:b])] for a, b in cuts)
    return BenchmarkDataset(
        name="lorenz",
        task=Regression(),
        train_loader=train,
        eval_loader=test,
        val_loader=val,
    )
