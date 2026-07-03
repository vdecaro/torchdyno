import importlib.util

import pytest

import torchdyno.data.datasets as datasets

torchvision_missing = importlib.util.find_spec("torchvision") is None
pandas_missing = importlib.util.find_spec("pandas") is None


@pytest.mark.skipif(not torchvision_missing, reason="torchvision is installed")
def test_seqmnist_missing_extra_message():
    with pytest.raises(ImportError, match=r"torchdyno\[datasets\]"):
        datasets.SequentialMNIST


@pytest.mark.skipif(not pandas_missing, reason="pandas is installed")
def test_hhar_missing_extra_message():
    with pytest.raises(ImportError, match=r"torchdyno\[datasets\]"):
        datasets.HHARDataset
