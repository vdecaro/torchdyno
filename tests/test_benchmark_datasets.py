import importlib.util

import pytest
import torch

from torchdyno import SequenceModel
from torchdyno.benchmark import BenchmarkDataset, BenchmarkSpec, run
from torchdyno.benchmark.datasets import (
    lorenz_benchmark,
    memory_capacity_benchmark,
    sequential_mnist_benchmark,
)
from torchdyno.heads import RegressionHead
from torchdyno.models.esn import ESNCore
from torchdyno.optim import RidgeRegression
from torchdyno.tasks import Regression, SequenceClassification

_TORCHVISION = importlib.util.find_spec("torchvision") is not None


def test_memory_capacity_provider_shapes_and_task():
    ds = memory_capacity_benchmark(delay=5, length=200, train_frac=0.5)
    assert isinstance(ds, BenchmarkDataset)
    assert isinstance(ds.task, Regression)
    (xtr, ytr), = ds.train_loader
    (xte, yte), = ds.eval_loader
    # (T, 1, 1) single-batch, split along time.
    assert xtr.shape[1:] == (1, 1) and ytr.shape[1:] == (1, 1)
    assert xtr.shape[0] + xte.shape[0] == 200 - 5


def test_lorenz_provider_shapes_and_task():
    ds = lorenz_benchmark(length=300, train_frac=0.5, target_delay=1)
    assert isinstance(ds.task, Regression)
    (xtr, ytr), = ds.train_loader
    assert xtr.shape[1] == 1 and xtr.shape[2] == 3  # (T, 1, 3)
    assert ytr.shape[2] == 3


def test_memory_capacity_end_to_end_beats_trivial():
    def build(task):
        core = ESNCore(input_size=1, layer_sizes=[100], input_scaling=0.9, rho=0.9)
        model = SequenceModel(core, RegressionHead(core.state_size, 1))
        return model, RidgeRegression(l2=1e-6, score_fn=task.primary, mode=task.primary.mode)

    ds = memory_capacity_benchmark(delay=3, length=1000, train_frac=0.5)
    result = run(BenchmarkSpec(ds, seeds=[0, 1, 2]), build)
    assert len(result.per_seed) == 3
    nrmse_mean, _ = result.metrics["nrmse"]
    assert nrmse_mean < 1.0  # a short-delay reservoir beats predicting the mean


def test_lorenz_end_to_end_runs():
    def build(task):
        core = ESNCore(input_size=3, layer_sizes=[100], input_scaling=0.9, rho=0.9)
        model = SequenceModel(core, RegressionHead(core.state_size, 3))
        return model, RidgeRegression(l2=1e-6, score_fn=task.primary, mode=task.primary.mode)

    ds = lorenz_benchmark(length=600, train_frac=0.5)
    result = run(BenchmarkSpec(ds, seeds=[0, 1]), build)
    assert torch.isfinite(torch.tensor(result.metrics["nrmse"][0]))


def test_sequential_mnist_provider_is_callable():
    # Importable + callable without constructing (no download).
    assert callable(sequential_mnist_benchmark)


@pytest.mark.skipif(_TORCHVISION, reason="torchvision is installed")
def test_sequential_mnist_gated_without_torchvision():
    with pytest.raises(ImportError, match=r"torchdyno\[datasets\]"):
        sequential_mnist_benchmark(root="/tmp/does-not-matter")
