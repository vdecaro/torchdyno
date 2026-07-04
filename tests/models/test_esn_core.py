import pytest
import torch

from torchdyno.models.base import CoreOutput, SequenceCore
from torchdyno.models.esn.core import ESNCore
from torchdyno.testing import assert_core_conforms


def test_esn_core_is_sequence_core():
    core = ESNCore(input_size=2, layer_sizes=[16])
    assert isinstance(core, SequenceCore)


def test_stacked_state_size_is_last_layer():
    core = ESNCore(input_size=2, layer_sizes=[16, 8], arch_type="stacked")
    assert core.state_size == 8


def test_multi_state_size_is_sum():
    core = ESNCore(input_size=2, layer_sizes=[16, 8], arch_type="multi")
    assert core.state_size == 24


def test_forward_shapes_and_capabilities():
    torch.manual_seed(0)
    core = ESNCore(input_size=2, layer_sizes=[16, 8], arch_type="stacked")
    x = torch.randn(10, 4, 2)
    out = core(x)
    assert isinstance(out, CoreOutput)
    assert out.states.shape == (10, 4, 8)
    assert out.layer_states is not None and len(out.layer_states) == 2
    assert out.layer_states[0].shape == (10, 4, 16)
    assert len(out.final_state) == 2


def test_conforms_stacked():
    torch.manual_seed(0)
    core = ESNCore(input_size=3, layer_sizes=[20, 12], arch_type="stacked")
    assert_core_conforms(core, torch.randn(8, 5, 3))


def test_conforms_multi():
    torch.manual_seed(0)
    core = ESNCore(input_size=3, layer_sizes=[20, 12], arch_type="multi")
    assert_core_conforms(core, torch.randn(8, 5, 3))


def test_rejects_empty_layers():
    with pytest.raises(ValueError):
        ESNCore(input_size=2, layer_sizes=[])


def test_rejects_bad_arch_type():
    with pytest.raises(ValueError):
        ESNCore(input_size=2, layer_sizes=[8], arch_type="nonsense")
