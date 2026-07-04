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


from torch import nn

from torchdyno import SequenceModel
from torchdyno.heads import RegressionHead
from torchdyno.optim import RidgeRegression, IntrinsicPlasticity
from torchdyno.training import BackpropTrainer
from torchdyno.training.base import FitResult


def _tiny_regression_loader(input_size, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(30, 4, input_size, generator=g)
    return [(x, x.clone())]


def _build_learner(name):
    if name == "ridge":
        return RidgeRegression(l2=1e-6)
    if name == "backprop":
        return BackpropTrainer(loss_fn=nn.MSELoss(), epochs=2)
    raise AssertionError(f"No minimal builder for learner {name!r}")


ADMIT_PAIRS = [(c, l) for c in CORE_NAMES for l in get_card(c).admits]


@pytest.mark.parametrize("core_name,learner_name", ADMIT_PAIRS)
def test_admitted_learner_runs(core_name, learner_name):
    torch.manual_seed(0)
    core = create_core(core_name)
    model = SequenceModel(core, RegressionHead(core.state_size, 1))
    learner = _build_learner(learner_name)
    result = learner.fit(model, _tiny_regression_loader(core.input_size))
    assert isinstance(result, FitResult)


def test_esn_admits_ip_adapter():
    torch.manual_seed(0)
    assert "ip" in get_card("esn").adapters
    core = create_core("esn", net_gain_and_bias=True)  # IP needs net_a/net_b
    res = core.reservoirs[0]
    res.train()
    ip = IntrinsicPlasticity(learning_rate=0.01, mu=0.0, sigma=0.1)
    ip.compile(res)
    x = torch.randn(20, 4, core.input_size)
    for _ in range(3):
        res(x)
        ip.backward()
        ip.step()
    ip.detach()
    assert not ip.compiled
