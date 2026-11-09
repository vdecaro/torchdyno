import torch

from torchdyno.benchmark import (
    lorenz_benchmark,
    lorenz_system,
)


def test_lorenz_system_is_normalized_and_deterministic():
    trajectory = lorenz_system(500)
    assert trajectory.shape == (500, 3)
    assert trajectory.abs().max() == 1.0
    assert torch.equal(trajectory, lorenz_system(500))


def test_lorenz_benchmark_splits_contiguously():
    data = lorenz_benchmark(length=400, target_delay=2, splits=(0.5, 0.25))
    ((x_train, y_train),) = data.train_loader
    ((x_val, _),) = data.val_loader
    ((x_test, _),) = data.eval_loader
    assert [x_train.shape[0], x_val.shape[0], x_test.shape[0]] == [200, 100, 100]
    assert x_train.shape[1:] == (1, 3)
    full = lorenz_system(402)
    assert torch.equal(x_train[:, 0], full[:200])
    assert torch.equal(y_train[:, 0], full[2:202])
    assert torch.equal(x_test[-1, 0], full[399])
