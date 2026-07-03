import importlib

import torchdyno.data.datasets as datasets


def test_core_only_datasets_import_without_extras():
    # These depend only on torch/numpy (core), so they must resolve.
    assert datasets.LorenzSystem is not None
    assert datasets.MemoryCapacityDataset is not None


def test_unknown_attribute_raises_attribute_error():
    try:
        datasets.NotADataset
    except AttributeError:
        return
    raise AssertionError("expected AttributeError for unknown attribute")


def test_dir_lists_public_datasets():
    names = dir(datasets)
    for expected in [
        "SequentialMNIST",
        "WESADDataset",
        "HHARDataset",
        "LorenzSystem",
        "MemoryCapacityDataset",
    ]:
        assert expected in names


def test_package_import_does_not_pull_optional_deps():
    # Importing the datasets package must not eagerly import the submodules
    # that require torchvision/pandas.
    importlib.reload(datasets)
    import sys

    assert "torchdyno.data.datasets.seq_mnist" not in sys.modules
    assert "torchdyno.data.datasets.hhar" not in sys.modules
