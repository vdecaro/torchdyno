import pytest

import torchdyno.optim as optim
from torchdyno.optim import IntrinsicPlasticity
from torchdyno.optim.adapters import CoreAdapter


def test_coreadapter_is_abstract():
    with pytest.raises(TypeError):
        CoreAdapter()  # abstract methods -> cannot instantiate


def test_intrinsic_plasticity_is_coreadapter():
    ip = IntrinsicPlasticity(learning_rate=0.01, mu=0.0, sigma=0.1)
    assert isinstance(ip, CoreAdapter)


def test_exports():
    assert hasattr(optim, "CoreAdapter")
    assert hasattr(optim, "IntrinsicPlasticity")
    assert optim.CoreAdapter is CoreAdapter
