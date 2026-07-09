from torchdyno.benchmark import BenchmarkResult, to_csv, to_markdown


def _result():
    return BenchmarkResult(
        dataset="toy",
        metrics={"nrmse": (0.25, 0.03), "mse": (0.1, 0.01)},
        per_seed=[{"nrmse": 0.22, "mse": 0.09}, {"nrmse": 0.28, "mse": 0.11}],
        seeds=[0, 1],
        wall_time_s=1.5,
        peak_memory_bytes=123456,
        config_hash="abc123def456",
    )


def test_to_markdown_covers_dataset_and_metrics():
    md = to_markdown(_result())
    assert isinstance(md, str) and md
    assert "toy" in md
    assert "abc123def456" in md
    for name in ("nrmse", "mse"):
        assert name in md


def test_to_csv_has_header_and_a_row_per_metric():
    csv = to_csv(_result())
    lines = csv.strip().splitlines()
    assert lines[0] == "metric,mean,std"
    assert len(lines) == 1 + 2  # header + one row per metric
    assert any(line.startswith("nrmse,") for line in lines)
    assert any(line.startswith("mse,") for line in lines)
