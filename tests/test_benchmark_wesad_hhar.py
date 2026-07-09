import importlib.util

import pytest
import torch

from torchdyno import BenchmarkSpec, SequenceModel, run
import torchdyno.benchmark.datasets as dsmod
from torchdyno.benchmark.datasets import hhar_benchmark, wesad_benchmark
from torchdyno.heads import DenseHead
from torchdyno.models.esn import ESNCore
from torchdyno.optim import RidgeRegression
from torchdyno.tasks import DenseLabeling

_PANDAS = importlib.util.find_spec("pandas") is not None


class _FakeDense(torch.utils.data.Dataset):
    """A stand-in dense dataset: one (seq_length, F) / (seq_length, C) one-hot item."""

    USERS = {"train": {50: ["2", "5"]}, "test": ["4"]}

    def __init__(self, root, user, context, seq_length=700):
        g = torch.Generator().manual_seed(abs(hash((user, context))) % 10_000)
        self._x = torch.randn(seq_length, 6, generator=g)
        labels = torch.randint(0, _FakeDense.n_classes, (seq_length,), generator=g)
        self._y = torch.nn.functional.one_hot(labels, _FakeDense.n_classes).float()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self._x, self._y


def test_providers_are_callable():
    assert callable(wesad_benchmark) and callable(hhar_benchmark)


def test_wesad_provider_wiring_and_dense_run(monkeypatch):
    _FakeDense.n_classes = 4
    monkeypatch.setattr(dsmod, "WESADDataset", _FakeDense)
    ds = wesad_benchmark(root="/unused", train_pct=50, seq_length=30, batch_size=2)
    assert isinstance(ds.task, DenseLabeling) and ds.task.num_classes == 4

    def build(task):
        core = ESNCore(input_size=6, layer_sizes=[32], input_scaling=0.9, rho=0.9)
        return SequenceModel(core, DenseHead(core.state_size, 4)), RidgeRegression(l2=1e-6)

    result = run(BenchmarkSpec(ds, seeds=[0]), build)
    acc = result.metrics["accuracy"][0]
    assert 0.0 <= acc <= 1.0  # dense multi-batch eval ran end-to-end


@pytest.mark.skipif(_PANDAS, reason="pandas is installed")
def test_hhar_gated_without_pandas():
    with pytest.raises(ImportError, match=r"torchdyno\[datasets\]"):
        hhar_benchmark(root="/unused")
