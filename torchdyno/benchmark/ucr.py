"""The UCR and UEA archives: time series classification from local files."""

from __future__ import annotations

from pathlib import Path

import torch

from torchdyno.benchmark.dataset import BenchmarkDataset
from torchdyno.objectives import SequenceClassification

Split = tuple[torch.Tensor, torch.Tensor]


def _read_tsv(path: Path) -> tuple[list[list[list[float]]], list[str]]:
    """UCR's format: a label, then the values, tab-separated; ``NaN`` pads."""
    series, labels = [], []
    for line in path.read_text().splitlines():
        if line.strip():
            label, *values = line.split("\t")
            series.append([[float(value)] for value in values])
            labels.append(label.strip())
    return series, labels


def _read_ts(path: Path) -> tuple[list[list[list[float]]], list[str]]:
    """UEA's ``.ts`` format: a header, then channels split by ``:``, label last."""
    series, labels, data = [], [], False
    for line in path.read_text().splitlines():
        line = line.strip()  # noqa: PLW2901
        if not line or line.startswith("#"):
            continue
        if not data:
            if line.lower().startswith("@timestamps true"):
                raise ValueError(f"{path} has timestamps, which are not supported.")
            data = line.lower() == "@data"
            continue
        *channels, label = line.split(":")
        values = [
            [float("nan") if v == "?" else float(v) for v in c.split(",")]
            for c in channels
        ]
        steps = max(map(len, values))
        padded = [c + [float("nan")] * (steps - len(c)) for c in values]
        series.append([list(step) for step in zip(*padded, strict=True)])
        labels.append(label.strip())
    return series, labels


def _sorted_classes(labels: set[str]) -> list[str]:
    """Labels in numeric order if they are all numbers, else in string order."""
    try:
        return sorted(labels, key=float)
    except ValueError:
        return sorted(labels)


def load_ucr(root: str | Path, name: str) -> tuple[Split, Split, list[str]]:
    """Read a UCR (``.tsv``) or UEA (``.ts``) dataset from its archive folder.

    Args:
        root: the archive folder, holding ``<name>/<name>_TRAIN`` and
            ``<name>/<name>_TEST`` with either extension.
        name: the dataset name, e.g. ``"GunPoint"``.

    Returns:
        The train and test splits, each ``(x, y)`` with time-first ``(T, N, C)``
        series, NaN past the end of shorter ones and at missing values, and
        ``(N,)`` class indices; and the class labels, by index.

    Raises:
        FileNotFoundError: if neither format is in ``root / name``.
    """
    folder = Path(root) / name
    suffix = next(
        (s for s in (".tsv", ".ts") if (folder / f"{name}_TRAIN{s}").exists()), None
    )
    if suffix is None:
        raise FileNotFoundError(f"No {name}_TRAIN.tsv or {name}_TRAIN.ts in {folder}.")
    read = _read_tsv if suffix == ".tsv" else _read_ts
    parts = [read(folder / f"{name}_{split}{suffix}") for split in ("TRAIN", "TEST")]
    classes = _sorted_classes({label for _, labels in parts for label in labels})
    index = {label: i for i, label in enumerate(classes)}
    steps = max(len(s) for series, _ in parts for s in series)
    splits = []
    for series, labels in parts:
        nan = [float("nan")] * len(series[0][0])
        x = torch.tensor([s + [nan] * (steps - len(s)) for s in series])
        splits.append((x.transpose(0, 1), torch.tensor([index[y] for y in labels])))
    return splits[0], splits[1], classes


def ucr_benchmark(
    root: str | Path,
    name: str,
    batch_size: int | None = None,
    val: float = 0.0,
) -> BenchmarkDataset:
    """A UCR or UEA dataset as a sequence classification benchmark.

    Args:
        root: the archive folder, see [`load_ucr()`][torchdyno.benchmark.load_ucr].
        name: the dataset name.
        batch_size: sequences per batch; ``None`` puts each split in one batch.
        val: the fraction of the training set held out for validation, drawn
            with torch's generator; ``0`` holds out none.

    Returns:
        The benchmark dataset, its loaders lists of ``(x, y)`` batches.
    """
    (x, y), (x_test, y_test), _ = load_ucr(root, name)

    def batches(x: torch.Tensor, y: torch.Tensor) -> list[Split]:
        size = batch_size or len(y)
        return [(x[:, i : i + size], y[i : i + size]) for i in range(0, len(y), size)]

    held = int(val * len(y))
    order = torch.randperm(len(y)) if held else torch.arange(len(y))
    train, rest = order[held:], order[:held]
    return BenchmarkDataset(
        name=name,
        task=SequenceClassification(),
        train_loader=batches(x[:, train], y[train]),
        eval_loader=batches(x_test, y_test),
        val_loader=batches(x[:, rest], y[rest]) if held else None,
    )
