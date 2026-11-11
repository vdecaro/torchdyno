import pytest
import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from tests.unit.compositions import adadiag
from torchdyno.benchmark import (
    BenchmarkDataset,
    BenchmarkSpec,
    run,
    to_markdown,
)
from torchdyno.learners import (
    Backprop,
    Ridge,
)
from torchdyno.pooling import Last
from torchdyno.tasks import Regression


def _dataset(val=True):
    torch.manual_seed(0)
    x = torch.randn(60, 2, 1)
    y = torch.roll(x, 1, dims=0)
    return BenchmarkDataset(
        name="toy",
        task=Regression(),
        train_loader=[(x[:30], y[:30])],
        eval_loader=[(x[45:], y[45:])],
        val_loader=[(x[30:45], y[30:45])] if val else None,
    )


def _build(task):
    # one random coupling pair: identical results need seeding
    model = nn.Sequential(adadiag(1, (8, 8), pairs=1), nn.Linear(16, 1))
    return model, Ridge([1e-6, 1e-2], score_fn=task.primary, mode="min")


def test_reports_mean_and_std_over_seeds():
    result = run(BenchmarkSpec(_dataset(), seeds=[0, 1, 2]), _build)
    assert result.seeds == [0, 1, 2]
    assert len(result.per_seed) == 3
    nrmse = [d["nrmse"] for d in result.per_seed]
    mean = sum(nrmse) / 3
    assert result.metrics["nrmse"][0] == pytest.approx(mean)
    assert result.metrics["nrmse"][1] == pytest.approx(
        torch.tensor(nrmse).std(unbiased=False).item()
    )


def test_identical_calls_give_identical_results():
    spec = BenchmarkSpec(_dataset(), seeds=[0, 1])
    assert run(spec, _build).per_seed == run(spec, _build).per_seed


def test_passes_the_validation_loader_to_the_learner():
    with pytest.raises(ValueError, match="needs val"):
        run(BenchmarkSpec(_dataset(val=False), seeds=[0]), _build)


def test_rejects_empty_seeds():
    with pytest.raises(ValueError, match="non-empty"):
        run(BenchmarkSpec(_dataset(), seeds=[]), _build)


def test_markdown_report():
    report = to_markdown(run(BenchmarkSpec(_dataset(), seeds=[0]), _build))
    assert report.startswith("# Benchmark: toy")
    assert "| nrmse |" in report


def test_scores_after_the_dataset_washout():
    class Unfitted:
        def fit(self, model, train, val):
            pass

    data = _dataset()
    data.washout = 5
    model = nn.Sequential(nn.Linear(1, 1))
    result = run(BenchmarkSpec(data, seeds=[0]), lambda task: (model, Unfitted()))
    x, y = data.eval_loader[0]
    with torch.no_grad():
        expected = torch.mean((model(x)[5:] - y[5:]) ** 2).item()
    assert result.per_seed[0]["mse"] == pytest.approx(expected)


def test_a_seed_gives_the_same_result_alone_or_in_a_sweep():
    x = torch.randn(32, 6, 1)
    pairs = TensorDataset(x, x.sum(1))

    def collate(batch):
        xs, ys = zip(*batch, strict=True)
        return torch.stack(xs, dim=1), torch.stack(ys)

    loader = DataLoader(
        pairs,
        batch_size=4,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator(),
    )
    data = BenchmarkDataset(
        "toy", Regression(), loader, [(x.transpose(0, 1), x.sum(1))]
    )

    def build(task):
        model = nn.Sequential(adadiag(1, (4, 4), pairs=1), Last(), nn.Linear(8, 1))
        return model, Backprop(task.loss, lr=1e-2, epochs=2)

    alone = run(BenchmarkSpec(data, seeds=[1]), build).per_seed
    assert run(BenchmarkSpec(data, seeds=[0, 1]), build).per_seed[1:] == alone
