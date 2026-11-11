import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from torchdyno.learners import QuantileBias
from torchdyno.readout import PPV
from torchdyno.sequence import (
    RandomConv,
    minirocket,
)


def test_bias_is_minus_a_golden_ratio_quantile_of_padded_outputs():
    torch.manual_seed(0)
    conv = RandomConv(1, length=40, kernels=84, draw=minirocket, dtype=torch.float64)
    model = nn.Sequential(conv, PPV())
    x = torch.randn(40, 1, 1, dtype=torch.float64)
    QuantileBias().fit(model, [(x, None)])
    for k in range(84):
        d = int(conv.dilations[k])
        out = F.conv1d(
            x.permute(1, 2, 0), conv.weight[k : k + 1], padding=4 * d, dilation=d
        )
        q = (k + 1) * (1 + math.sqrt(5)) / 2 % 1
        expected = -torch.quantile(out.flatten(), q)
        torch.testing.assert_close(conv.bias[k], expected)
    padded = conv.paddings > 0
    ppv = model(x)[0, padded]
    q = (torch.arange(1, 85, dtype=torch.float64) * (1 + math.sqrt(5)) / 2 % 1)[padded]
    assert (ppv - (1 - q)).abs().max() <= 1 / 40


def test_skips_absent_steps_and_needs_one_random_conv():
    torch.manual_seed(0)
    conv = RandomConv(1, length=20, kernels=84, draw=minirocket)
    x = torch.randn(30, 4, 1)
    x[20:] = torch.nan
    QuantileBias().fit(nn.Sequential(conv), [(x, None)])
    assert conv.bias.isfinite().all()
    with pytest.raises(TypeError, match="one RandomConv"):
        QuantileBias().fit(nn.Sequential(nn.Identity()), [(x, None)])
