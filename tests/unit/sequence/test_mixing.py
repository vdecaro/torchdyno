"""Mix against the complex convolution it stands for."""

import pytest
import torch
import torch.nn.functional as F

from torchdyno.sequence import Mix

DTYPE = torch.float64


def test_mix_is_the_real_part_of_a_complex_convolution_across_units():
    torch.manual_seed(0)
    mix = Mix(channels=2, kernel_size=3, dtype=DTYPE)
    h = torch.randn(7, 4, 10, dtype=torch.complex128)
    weight, bias = mix.conv.weight[0].detach(), mix.conv.bias.detach()
    w = torch.complex(weight[0], -weight[1])
    padded = F.pad(h, (1, 1))
    expected = sum(w[j] * padded[..., j : j + 10] for j in range(3)).real + bias
    torch.testing.assert_close(mix(torch.cat([h.real, h.imag], -1)), expected)


def test_mix_keeps_absent_steps_absent():
    x = torch.randn(5, 2, 8)
    x[3, 1] = torch.nan
    out = Mix(kernel_size=5)(x)
    assert out.shape == (5, 2, 8)
    assert out[3, 1].isnan().all()
    assert not out[3, 0].isnan().any() and not out[:3].isnan().any()


def test_mix_rejects_features_that_do_not_split_into_its_channels():
    with pytest.raises(ValueError, match="7 features"):
        Mix(channels=2)(torch.randn(3, 2, 7))
