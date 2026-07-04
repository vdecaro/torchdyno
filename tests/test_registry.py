import pytest
import torch

import torchdyno  # noqa: F401  (populates the registry)
from torchdyno import (
    create_core,
    create_learner,
    create_head,
    list_cores,
    list_learners,
    list_heads,
    get_card,
    render_catalog,
)
from torchdyno.models.base import SequenceCore
from torchdyno.testing import assert_core_conforms

CORE_NAMES = ["esn", "scn", "adadiag"]


def test_cores_registered():
    assert set(CORE_NAMES).issubset(list_cores())


def test_learners_and_heads_registered():
    assert {"ridge", "backprop"}.issubset(list_learners())
    assert {"regression", "classification"}.issubset(list_heads())


@pytest.mark.parametrize("name", CORE_NAMES)
def test_create_core_builds_and_conforms(name):
    torch.manual_seed(0)
    core = create_core(name)
    assert isinstance(core, SequenceCore)
    x = torch.randn(6, 2, core.input_size)
    assert_core_conforms(core, x)


@pytest.mark.parametrize("name", CORE_NAMES)
def test_card_wellformed(name):
    card = get_card(name)
    assert card.name == name
    assert card.family and card.paper and card.description
    assert card.tasks
    assert isinstance(card.default_config, dict) and card.default_config
    for learner_name in card.admits:
        assert learner_name in list_learners()
    for adapter_name in card.adapters:
        assert isinstance(adapter_name, str) and adapter_name


def test_create_core_overrides_win():
    assert create_core("esn", layer_sizes=[8]).state_size == 8


def test_unknown_names_raise():
    with pytest.raises(KeyError):
        create_core("nope")
    with pytest.raises(KeyError):
        create_learner("nope")
    with pytest.raises(KeyError):
        create_head("nope")


def test_render_catalog_covers_all_cores():
    md = render_catalog()
    assert isinstance(md, str) and md
    for name in CORE_NAMES:
        assert name in md
