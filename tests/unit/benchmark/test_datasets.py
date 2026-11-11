import pytest
import torch

from torchdyno.benchmark import (
    load_ucr,
    lorenz_benchmark,
    lorenz_system,
    mackey_glass,
    mackey_glass_benchmark,
    memory_capacity_benchmark,
    narma,
    narma_benchmark,
    ucr_benchmark,
)
from torchdyno.objectives import MemoryCapacity


def test_lorenz_system_is_normalized_and_deterministic():
    trajectory = lorenz_system(500)
    assert trajectory.shape == (500, 3)
    assert trajectory.abs().max() == 1.0
    assert torch.equal(trajectory, lorenz_system(500))


def test_lorenz_benchmark_splits_contiguously():
    data = lorenz_benchmark(length=400, target_delay=2, splits=(0.5, 0.25))
    ((x_train, y_train),) = data.train_loader
    ((x_val, _),) = data.val_loader
    ((x_test, _),) = data.eval_loader
    assert [x_train.shape[0], x_val.shape[0], x_test.shape[0]] == [200, 100, 100]
    assert x_train.shape[1:] == (1, 3)
    full = lorenz_system(402)
    assert torch.equal(x_train[:, 0], full[:200])
    assert torch.equal(y_train[:, 0], full[2:202])
    assert torch.equal(x_test[-1, 0], full[399])


def test_mackey_glass_takes_euler_steps_from_a_constant_history():
    first = mackey_glass(2, step=0.1, sample=0.1)[0].item()
    assert first == pytest.approx(1.2 + 0.1 * (0.2 * 1.2 / (1 + 1.2**10) - 0.1 * 1.2))
    series = mackey_glass(1000)
    assert 0.2 < series.min() < series.max() < 1.4
    data = mackey_glass_benchmark(length=400, target_delay=3)
    ((x, y),) = data.train_loader
    assert x.shape == (200, 1, 1) and torch.equal(x[3:], y[:-3])


def test_narma_follows_its_recurrence():
    torch.manual_seed(0)
    u = torch.rand(30, dtype=torch.float64) * 0.5
    y = narma(u, order=4)
    assert (y[:3] == 0).all()
    assert y[3].item() == pytest.approx((1.5 * u[0] * u[3] + 0.1).item())
    expected = 0.3 * y[3] + 0.05 * y[3] * y[3] + 1.5 * u[1] * u[4] + 0.1
    assert y[4].item() == pytest.approx(expected.item())
    ((x, _),) = narma_benchmark(length=200).train_loader
    assert x.shape == (100, 1, 1)
    assert x.min() >= 0 and x.max() <= 0.5


def test_memory_capacity_targets_are_the_delayed_inputs():
    data = memory_capacity_benchmark(length=400, delays=5)
    ((x, y),) = data.train_loader
    assert y.shape == (200, 1, 5) and data.washout == 5
    for k in range(1, 6):
        assert torch.equal(y[k:, 0, k - 1], x[:-k, 0, 0])
    assert isinstance(data.task.primary, MemoryCapacity)


def test_ucr_and_uea_files_load_time_first_and_padded_with_nan(tmp_path):
    folder = tmp_path / "Toy"
    folder.mkdir()
    (folder / "Toy_TRAIN.tsv").write_text(
        "2\t1.0\t2.0\t3.0\n10\t4.0\t5.0\tNaN\n2\t0\t0\t0\n"
    )
    (folder / "Toy_TEST.tsv").write_text("10\t1.0\t2.0\t3.0\n")
    (x, y), (x_test, y_test), classes = load_ucr(tmp_path, "Toy")
    assert classes == ["2", "10"]
    assert x.shape == (3, 3, 1) and y.tolist() == [0, 1, 0] and y_test.tolist() == [1]
    assert x[:, 1, 0].tolist()[:2] == [4.0, 5.0] and x[2, 1, 0].isnan()
    folder = tmp_path / "Multi"
    folder.mkdir()
    header = "# a comment\n@problemName Multi\n@classLabel true a b\n@data\n"
    (folder / "Multi_TRAIN.ts").write_text(header + "1,2,3:4,5,6:a\n7,?:8,9:b\n")
    (folder / "Multi_TEST.ts").write_text(header + "1,2:3,4:b\n")
    (x, y), (x_test, _), classes = load_ucr(tmp_path, "Multi")
    assert classes == ["a", "b"] and y.tolist() == [0, 1]
    assert x.shape == (3, 2, 2) and x_test.shape == (3, 1, 2)
    assert x[:, 0].tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert x[1, 1, 0].isnan() and x[2, 1].isnan().all()
    data = ucr_benchmark(tmp_path, "Toy", batch_size=2, val=1 / 3)
    assert [len(b[1]) for b in data.train_loader] == [2]
    assert len(data.val_loader[0][1]) == 1
    with pytest.raises(FileNotFoundError):
        load_ucr(tmp_path, "Missing")
