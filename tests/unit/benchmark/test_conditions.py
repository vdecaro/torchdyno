import pytest
import torch
from torch import nn

from tests.unit.compositions import adadiag
from torchdyno.benchmark import (
    CONDITIONS,
    BenchmarkDataset,
    BenchmarkSpec,
    conditions_table,
    run_conditions,
)
from torchdyno.pooling import Mean
from torchdyno.tasks import SequenceClassification


def _spec():
    torch.manual_seed(0)
    x = torch.randn(15, 48, 2)
    y = (x[..., 0].mean(dim=0) > 0).long()
    split = [(x[:, :32], y[:32])], [(x[:, 32:40], y[32:40])], [(x[:, 40:], y[40:])]
    data = BenchmarkDataset(
        "toy", SequenceClassification(), split[0], split[2], split[1]
    )
    return BenchmarkSpec(data, seeds=[0, 1])


def _make_model():
    return nn.Sequential(adadiag(2, (8, 8), pairs=1), Mean(), nn.Linear(16, 2))


def _run(make_model=_make_model):
    return run_conditions(_spec(), make_model, l2=[1e-4, 1e-1], lr=1e-2, epochs=3)


def test_runs_every_condition_on_the_same_seeds():
    results = _run()
    assert tuple(results) == CONDITIONS
    assert all(r.seeds == [0, 1] for r in results.values())
    assert all(set(r.metrics) == {"accuracy"} for r in results.values())


def test_identical_calls_give_identical_results():
    first, second = _run(), _run()
    assert {c: r.per_seed for c, r in first.items()} == {
        c: r.per_seed for c, r in second.items()
    }


def test_each_condition_trains_only_its_parts():
    built = []

    def make_model():
        model = _make_model()
        built.append((model, {k: v.clone() for k, v in model.state_dict().items()}))
        return model

    _run(make_model)
    readout, frozen, trained = built[0], built[2], built[4]  # seed 0 of each

    def changed(entry):
        model, start = entry
        state = model.state_dict()
        return {k.split(".")[0] for k in start if not torch.equal(state[k], start[k])}

    assert changed(readout) == {"2"}
    assert changed(frozen) == {"2"}
    assert changed(trained) == {"0", "2"}
    assert torch.equal(
        readout[1]["0.cell.input.weight"], trained[1]["0.cell.input.weight"]
    )


def test_table_has_a_row_per_condition():
    table = conditions_table(_run()).splitlines()
    assert table[0] == "| condition | accuracy | wall time (s) |"
    assert [line.split(" | ")[0] for line in table[2:]] == [
        f"| {c}" for c in CONDITIONS
    ]
    assert " ± " in table[2]


def test_passes_ridge_options_to_the_readout_condition():
    def conditions(ridge):
        return run_conditions(
            _spec(), _make_model, l2=1e-4, lr=1e-2, epochs=1, ridge=ridge
        )

    assert set(conditions({"standardize": True})) == set(CONDITIONS)
    with pytest.raises(TypeError, match="unexpected keyword"):
        conditions({"standardise": True})
