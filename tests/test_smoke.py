import torch

import torchdyno
from torchdyno import SequenceModel
from torchdyno.models.esn import ESNCore
from torchdyno.heads import RegressionHead


def test_version_is_nonempty_string():
    assert isinstance(torchdyno.__version__, str)
    assert torchdyno.__version__ != ""


def test_esn_forward_shape_is_time_first():
    torch.manual_seed(0)
    core = ESNCore(
        input_size=1,
        layer_sizes=[32],
        arch_type="stacked",
        rho=0.9,
        input_scaling=0.5,
    )
    model = SequenceModel(core, RegressionHead(input_size=32, output_size=1))
    x = torch.randn(20, 4, 1)  # (T, B, F) — time-first convention
    y = model(x)
    assert y.shape == (20, 4, 1)
