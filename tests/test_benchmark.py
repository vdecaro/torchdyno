import pytest
import torch

from torchdyno import SequenceModel
from torchdyno.benchmark import BenchmarkDataset, BenchmarkResult, BenchmarkSpec, run
from torchdyno.heads import RegressionHead
from torchdyno.models.esn import ESNCore
from torchdyno.optim import RidgeRegression
from torchdyno.tasks import Regression


def _toy_dataset():
    # A tiny fixed regression: reconstruct the input. Single-batch (T, 1, 1).
    torch.manual_seed(0)
    x = torch.randn(120, 1, 1)
    train = [(x[:80], x[:80].clone())]
    eval_ = [(x[80:], x[80:].clone())]
    return BenchmarkDataset(
        name="toy", task=Regression("nrmse"), train_loader=train, eval_loader=eval_
    )


def _build(task):
    core = ESNCore(input_size=1, layer_sizes=[50], input_scaling=0.9, rho=0.9)
    model = SequenceModel(core, RegressionHead(core.state_size, 1))
    learner = RidgeRegression(l2=1e-6, score_fn=task.primary, mode=task.primary.mode)
    return model, learner


def test_run_returns_result_with_per_seed_and_aggregates():
    spec = BenchmarkSpec(dataset=_toy_dataset(), seeds=[0, 1, 2])
    result = run(spec, _build)
    assert isinstance(result, BenchmarkResult)
    assert result.dataset == "toy"
    assert len(result.per_seed) == 3
    # Every metric of the task appears, as (mean, std).
    for name in ("nrmse", "mse", "mae"):
        assert name in result.metrics
        mean, std = result.metrics[name]
        assert isinstance(mean, float) and isinstance(std, float)
    assert result.seeds == [0, 1, 2]
    assert result.wall_time_s >= 0.0
    assert result.peak_memory_bytes is not None
    assert isinstance(result.config_hash, str) and len(result.config_hash) == 12


def test_run_is_deterministic():
    spec = BenchmarkSpec(dataset=_toy_dataset(), seeds=[0, 1])
    r1 = run(spec, _build)
    r2 = run(spec, _build)
    assert r1.per_seed == r2.per_seed


def test_single_seed_std_is_zero():
    spec = BenchmarkSpec(dataset=_toy_dataset(), seeds=[7])
    result = run(spec, _build)
    assert all(std == 0.0 for _, std in result.metrics.values())


def test_config_hash_changes_with_seeds():
    ds = _toy_dataset()
    h1 = run(BenchmarkSpec(ds, [0, 1]), _build).config_hash
    h2 = run(BenchmarkSpec(ds, [0, 1, 2]), _build).config_hash
    assert h1 != h2


def test_run_empty_seeds_raises():
    with pytest.raises(ValueError, match="seeds"):
        run(BenchmarkSpec(dataset=_toy_dataset(), seeds=[]), _build)
