import pytest

from torchdyno import BenchmarkSpec, SequenceModel
from torchdyno.benchmark import memory_capacity_benchmark, search, SearchResult
from torchdyno.heads import RegressionHead
from torchdyno.models.esn import ESNCore
from torchdyno.optim import RidgeRegression


def _build_from_config(cfg):
    def build(task):
        core = ESNCore(input_size=1, layer_sizes=[80], input_scaling=0.9, rho=cfg["rho"])
        model = SequenceModel(core, RegressionHead(core.state_size, 1))
        learner = RidgeRegression(l2=1e-6, score_fn=task.primary, mode=task.primary.mode)
        return model, learner

    return build


def _spec():
    ds = memory_capacity_benchmark(delay=5, length=1000, train_frac=0.5)
    return BenchmarkSpec(ds, seeds=[0, 1])


def test_grid_search_selects_best_nrmse():
    res = search(
        {"rho": [0.3, 0.9]}, _build_from_config, _spec(),
        objective="nrmse", mode="min", strategy="grid",
    )
    assert isinstance(res, SearchResult)
    assert len(res.trials) == 2
    assert res.best_score == min(s for _, s in res.trials)
    assert res.best_config["rho"] in (0.3, 0.9)
    assert res.best_result.config_hash  # a real BenchmarkResult


def test_random_search_is_deterministic_and_sized():
    space = {"rho": [0.3, 0.6, 0.9]}
    r1 = search(space, _build_from_config, _spec(), objective="nrmse",
                strategy="random", n_samples=3, seed=0)
    r2 = search(space, _build_from_config, _spec(), objective="nrmse",
                strategy="random", n_samples=3, seed=0)
    assert len(r1.trials) == 3
    assert [c for c, _ in r1.trials] == [c for c, _ in r2.trials]


def test_mode_max_selects_largest():
    res = search({"rho": [0.3, 0.9]}, _build_from_config, _spec(),
                 objective="nrmse", mode="max", strategy="grid")
    assert res.best_score == max(s for _, s in res.trials)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        search({"rho": [0.3]}, _build_from_config, _spec(),
               objective="nrmse", strategy="nope")
