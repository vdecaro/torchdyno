import pytest
import torch
from torch import nn

from tests.unit.compositions import (
    lru,
    s4d,
)
from torchdyno.learners import Ridge
from torchdyno.readout import Mean
from torchdyno.sequence import (
    Recurrent,
    Scan,
)
from torchdyno.testing import assert_layer_conforms


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("build", [lru, s4d])
def test_scan_conforms(build, dtype, frozen):
    torch.manual_seed(0)
    layer = Scan(build(dtype=dtype).cell).requires_grad_(not frozen)
    assert_layer_conforms(layer, torch.randn(9, 3, 2, dtype=dtype))


@pytest.mark.parametrize(
    ("dtype", "atol"), [(torch.float64, 1e-12), (torch.float32, 1e-5)]
)
@pytest.mark.parametrize("build", [lru, s4d])
def test_scan_equals_the_loop_over_long_sequences(build, dtype, atol):
    torch.manual_seed(0)
    cell = build(dtype=dtype).cell
    x = torch.randn(300, 2, 2, dtype=dtype)
    h0 = torch.randn(2, cell.state_size, dtype=dtype)
    torch.testing.assert_close(
        Scan(cell)(x, h0), Recurrent(cell)(x, h0), rtol=0, atol=atol
    )


def test_ridge_reads_the_real_and_imaginary_parts():
    torch.manual_seed(0)
    model = nn.Sequential(
        Scan(lru(features=1, modes=16).cell), Mean(), nn.Linear(32, 2)
    )
    x = torch.randn(30, 640, 1)
    labels = (x[..., 0].mean(0) > 0).long()
    Ridge(l2=1e-4).fit(model, [(x[:, :512], labels[:512])])
    assert (model(x[:, 512:]).argmax(-1) == labels[512:]).float().mean() > 0.9
