import pytest
import torch
from torch import Tensor
from typing import Any, Optional

import torchdyno.registry as reg
from torchdyno.registry import (
    ModelCard,
    register_core,
    register_learner,
    register_head,
    create_core,
    create_learner,
    create_head,
    list_cores,
    list_learners,
    list_heads,
    get_card,
    render_catalog,
)
from torchdyno.models.base import CoreCapabilities, CoreOutput, SequenceCore


@pytest.fixture
def clean_registry():
    """Snapshot the global registry dicts and restore them after the test."""
    saved = (dict(reg._CORES), dict(reg._LEARNERS), dict(reg._HEADS))
    yield
    reg._CORES.clear(); reg._CORES.update(saved[0])
    reg._LEARNERS.clear(); reg._LEARNERS.update(saved[1])
    reg._HEADS.clear(); reg._HEADS.update(saved[2])


class _DummyCore(SequenceCore):
    def __init__(self, input_size: int = 1, state_size: int = 3):
        super().__init__()
        self.input_size = input_size
        self.state_size = state_size
        self.linear = torch.nn.Linear(input_size, state_size)
        self.capabilities = CoreCapabilities(
            compute_mode="loop",
            differentiable=True,
            trainable_recurrence=True,
            supports_step=False,
            admits_feedback=False,
            exposes_layer_states=False,
            dtype=torch.float32,
        )

    def forward(self, x: Tensor, *, state0: Any = None, mask: Optional[Tensor] = None) -> CoreOutput:
        states = self.linear(x)
        return CoreOutput(states=states, final_state=states[-1])


class _DummyLearner:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


_DUMMY_CARD = ModelCard(
    name="dummy",
    family="test",
    paper="A. Author. Dummy paper. 2026.",
    description="A dummy core for registry tests.",
    admits=("dummy_learner",),
    adapters=(),
    tasks=("forecast",),
    default_config={"input_size": 1, "state_size": 3},
)


def test_register_and_create_core(clean_registry):
    register_core("dummy", card=_DUMMY_CARD)(_DummyCore)
    core = create_core("dummy")
    assert isinstance(core, _DummyCore)
    assert core.state_size == 3
    assert create_core("dummy", state_size=5).state_size == 5  # override wins
    assert "dummy" in list_cores()
    assert get_card("dummy") is _DUMMY_CARD


def test_register_and_create_learner_and_head(clean_registry):
    register_learner("dummy_learner")(_DummyLearner)
    register_head("dummy_head")(_DummyLearner)
    assert isinstance(create_learner("dummy_learner", a=1), _DummyLearner)
    assert create_learner("dummy_learner", a=1).kwargs == {"a": 1}
    assert isinstance(create_head("dummy_head"), _DummyLearner)
    assert "dummy_learner" in list_learners()
    assert "dummy_head" in list_heads()


def test_unknown_names_raise(clean_registry):
    with pytest.raises(KeyError):
        create_core("nope")
    with pytest.raises(KeyError):
        create_learner("nope")
    with pytest.raises(KeyError):
        create_head("nope")


def test_duplicate_distinct_registration_raises(clean_registry):
    register_core("dummy", card=_DUMMY_CARD)(_DummyCore)
    register_core("dummy", card=_DUMMY_CARD)(_DummyCore)  # same class: idempotent
    class _Other(_DummyCore):
        pass
    with pytest.raises(ValueError):
        register_core("dummy", card=_DUMMY_CARD)(_Other)  # different class: error


def test_card_name_mismatch_raises(clean_registry):
    bad = ModelCard(
        name="other", family="test", paper="x", description="x",
        admits=(), adapters=(), tasks=("forecast",), default_config={},
    )
    with pytest.raises(ValueError):
        register_core("dummy", card=bad)(_DummyCore)


def test_render_catalog_mentions_registered_core(clean_registry):
    register_core("dummy", card=_DUMMY_CARD)(_DummyCore)
    md = render_catalog()
    assert isinstance(md, str) and "dummy" in md
    assert "compute_mode=loop" in md  # capabilities read from a built instance
