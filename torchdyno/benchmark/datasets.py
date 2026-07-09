"""Dataset providers: build a BenchmarkDataset with fixed train/eval splits."""

from typing import List, Literal, Optional

import torch
from torch.utils.data import ConcatDataset, DataLoader

from torchdyno.benchmark.dataset import BenchmarkDataset
from torchdyno.data.datasets import LorenzSystem, MemoryCapacityDataset, WESADDataset
from torchdyno.data.utils import seq_collate_fn
from torchdyno.tasks import DenseLabeling, Regression, SequenceClassification


def _single_batch_loader(x: torch.Tensor, y: torch.Tensor):
    """A one-batch loader of a time-first ``(T, 1, F)`` pair."""
    return [(x, y)]


def memory_capacity_benchmark(
    delay: int = 10, length: int = 6000, train_frac: float = 0.5, seed: int = 0
) -> BenchmarkDataset:
    """Memory-capacity forecasting: input ``u_t``, target ``u_{t-delay}``."""
    data = MemoryCapacityDataset(delay=delay, length=length, seed=seed).data
    x = data[delay:].reshape(-1, 1, 1)          # (T, 1, 1) inputs u_t
    y = data[: length - delay].reshape(-1, 1, 1)  # (T, 1, 1) targets u_{t-delay}
    n_train = int(x.shape[0] * train_frac)
    return BenchmarkDataset(
        name="memory_capacity",
        task=Regression("nrmse"),
        train_loader=_single_batch_loader(x[:n_train], y[:n_train]),
        eval_loader=_single_batch_loader(x[n_train:], y[n_train:]),
    )


def lorenz_benchmark(
    length: int = 2000,
    train_frac: float = 0.5,
    target_delay: int = 1,
    input_dimensions: Optional[List[Literal["x", "y", "z"]]] = None,
) -> BenchmarkDataset:
    """Lorenz-system forecasting: predict the state ``target_delay`` steps ahead."""
    ds = LorenzSystem(
        length=length,
        target_delay=target_delay,
        input_dimensions=input_dimensions,
        return_full_sequence=True,
    )
    x_full, y_full = ds[0]                 # (T, n_in), (T, 3)
    x = x_full.unsqueeze(1)               # (T, 1, n_in)
    y = y_full.unsqueeze(1)               # (T, 1, 3)
    n_train = int(x.shape[0] * train_frac)
    return BenchmarkDataset(
        name="lorenz",
        task=Regression("nrmse"),
        train_loader=_single_batch_loader(x[:n_train], y[:n_train]),
        eval_loader=_single_batch_loader(x[n_train:], y[n_train:]),
    )


def sequential_mnist_benchmark(
    root: str,
    batch_size: int = 64,
    download: bool = False,
    permute_seed: Optional[int] = None,
) -> BenchmarkDataset:
    """Row-by-row sequential MNIST classification (needs the ``[datasets]`` extra)."""
    from torch.utils.data import DataLoader

    from torchdyno.data.datasets import SequentialMNIST  # torchvision-gated import

    collate = seq_collate_fn()
    train = SequentialMNIST(
        root=root, train=True, download=download, permute_seed=permute_seed
    )
    test = SequentialMNIST(
        root=root, train=False, download=download, permute_seed=permute_seed
    )
    return BenchmarkDataset(
        name="sequential_mnist",
        task=SequenceClassification(num_classes=10),
        train_loader=DataLoader(train, batch_size=batch_size, collate_fn=collate),
        eval_loader=DataLoader(test, batch_size=batch_size, collate_fn=collate),
    )


def wesad_benchmark(
    root: str,
    train_pct: int = 50,
    context: int = 0,
    seq_length: int = 700,
    batch_size: int = 32,
) -> BenchmarkDataset:
    """WESAD per-timestep stress classification (4 classes) over user groups."""
    users = WESADDataset.USERS
    train = ConcatDataset(
        [
            WESADDataset(root=root, user=u, context=context, seq_length=seq_length)
            for u in users["train"][train_pct]
        ]
    )
    ev = ConcatDataset(
        [
            WESADDataset(root=root, user=u, context=context, seq_length=seq_length)
            for u in users["test"]
        ]
    )
    collate = seq_collate_fn()
    return BenchmarkDataset(
        name="wesad",
        task=DenseLabeling(num_classes=4),
        train_loader=DataLoader(train, batch_size=batch_size, collate_fn=collate),
        eval_loader=DataLoader(ev, batch_size=batch_size, collate_fn=collate),
    )


def hhar_benchmark(
    root: str,
    train_pct: int = 50,
    context: int = 0,
    seq_length: int = 200,
    batch_size: int = 32,
) -> BenchmarkDataset:
    """HHAR per-timestep activity classification (6 classes) — needs the ``[datasets]`` extra."""
    from torchdyno.data.datasets import HHARDataset  # pandas-gated import

    users = HHARDataset.USERS
    train = ConcatDataset(
        [
            HHARDataset(root=root, user=u, context=context, seq_length=seq_length)
            for u in users["train"][train_pct]
        ]
    )
    ev = ConcatDataset(
        [
            HHARDataset(root=root, user=u, context=context, seq_length=seq_length)
            for u in users["test"]
        ]
    )
    collate = seq_collate_fn()
    return BenchmarkDataset(
        name="hhar",
        task=DenseLabeling(num_classes=6),
        train_loader=DataLoader(train, batch_size=batch_size, collate_fn=collate),
        eval_loader=DataLoader(ev, batch_size=batch_size, collate_fn=collate),
    )
