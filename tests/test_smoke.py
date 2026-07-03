import torch

import torchdyno
from torchdyno.models.esn import EchoStateNetwork


def test_version_is_nonempty_string():
    assert isinstance(torchdyno.__version__, str)
    assert torchdyno.__version__ != ""


def test_esn_forward_shape_is_time_first():
    torch.manual_seed(0)
    model = EchoStateNetwork(
        output_size=1,
        input_size=1,
        layer_sizes=[32],
        rho=0.9,
        input_scaling=0.5,
    )
    x = torch.randn(20, 4, 1)  # (T, B, F) — time-first convention
    y = model(x)
    assert y.shape == (20, 4, 1)
