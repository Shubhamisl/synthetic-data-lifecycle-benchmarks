from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmarks import benchmark_models
from benchmarks.run_benchmarks import _configure_stdout_utf8
from benchmarks.run_benchmarks import _synthetic_files, main


def test_configure_stdout_utf8_reconfigures_when_supported(monkeypatch):
    calls: list[dict[str, str]] = []

    class DummyStdout:
        def reconfigure(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr("sys.stdout", DummyStdout())

    _configure_stdout_utf8()

    assert calls == [{"encoding": "utf-8"}]


def test_synthetic_files_follow_registry_model_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_models, "get_trainable_benchmark_model_ids", lambda: ("ctgan", "tabddpm"))

    def fake_get_dataset_paths(dataset: str) -> dict[str, Path]:
        return {
            "train": tmp_path / f"{dataset}_train.csv",
            "test": tmp_path / f"{dataset}_test.csv",
            "ctgan": tmp_path / f"{dataset}_ctgan.csv",
            "tvae": tmp_path / f"{dataset}_tvae.csv",
            "tabddpm": tmp_path / f"{dataset}_tabddpm.csv",
        }

    monkeypatch.setattr("benchmarks.run_benchmarks.get_dataset_paths", fake_get_dataset_paths)

    paths = _synthetic_files()

    assert paths == [
        tmp_path / "bank_ctgan.csv",
        tmp_path / "bank_tabddpm.csv",
        tmp_path / "covertype_ctgan.csv",
        tmp_path / "covertype_tabddpm.csv",
        tmp_path / "diabetes_ctgan.csv",
        tmp_path / "diabetes_tabddpm.csv",
    ]


def test_main_uses_registry_model_ids_for_training_gate(tmp_path, monkeypatch):
    benchmark_root = tmp_path / "benchmarks"
    datasets_dir = benchmark_root / "datasets"
    results_dir = benchmark_root / "results"
    plots_dir = benchmark_root / "plots"
    for path in (datasets_dir, results_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("benchmarks.run_benchmarks.BENCHMARK_ROOT", benchmark_root)
    monkeypatch.setattr("benchmarks.common.BENCHMARK_ROOT", benchmark_root)
    monkeypatch.setattr("benchmarks.visualize_benchmarks.SUMMARY_PATH", results_dir / "summary.csv")
    monkeypatch.setattr("benchmarks.visualize_benchmarks.MEAN_RANK_PATH", results_dir / "mean_rank.csv")
    monkeypatch.setattr("benchmarks.run_benchmarks.SUMMARY_PATH", results_dir / "summary.csv")
    monkeypatch.setattr("benchmarks.run_benchmarks.MEAN_RANK_PATH", results_dir / "mean_rank.csv")
    monkeypatch.setattr("benchmarks.run_benchmarks.required_plot_paths", lambda: [])
    monkeypatch.setattr("benchmarks.run_benchmarks._configure_stdout_utf8", lambda: None)

    monkeypatch.setattr(
        benchmark_models,
        "get_trainable_benchmark_model_ids",
        lambda: ("ctgan", "tvae", "tabddpm"),
    )

    def fake_get_dataset_paths(dataset: str) -> dict[str, Path]:
        return {
            "train": datasets_dir / f"{dataset}_train.csv",
            "test": datasets_dir / f"{dataset}_test.csv",
            "ctgan": benchmark_root / "synthetic" / f"{dataset}_ctgan.csv",
            "tvae": benchmark_root / "synthetic" / f"{dataset}_tvae.csv",
            "tabddpm": benchmark_root / "synthetic" / f"{dataset}_tabddpm.csv",
        }

    monkeypatch.setattr("benchmarks.run_benchmarks.get_dataset_paths", fake_get_dataset_paths)

    for dataset in ("adult", "bank", "covertype", "diabetes"):
        paths = fake_get_dataset_paths(dataset)
        paths["train"].parent.mkdir(parents=True, exist_ok=True)
        paths["train"].write_text("x\n1\n", encoding="utf-8")
        paths["test"].write_text("x\n1\n", encoding="utf-8")
    for dataset in ("bank", "covertype", "diabetes"):
        paths = fake_get_dataset_paths(dataset)
        paths["ctgan"].parent.mkdir(parents=True, exist_ok=True)
        paths["ctgan"].write_text("x\n1\n", encoding="utf-8")
        paths["tvae"].write_text("x\n1\n", encoding="utf-8")

    pd.DataFrame([{"model": "ctgan"}]).to_csv(results_dir / "summary.csv", index=False)
    pd.DataFrame([{"rank": 1}]).to_csv(results_dir / "mean_rank.csv", index=False)

    modules: list[str] = []

    def fake_run_module(module_name: str) -> None:
        modules.append(module_name)

    monkeypatch.setattr("benchmarks.run_benchmarks._run_module", fake_run_module)

    main()

    assert "benchmarks.train_benchmark_models" in modules


def test_main_triggers_training_when_registry_default_model_output_is_missing(tmp_path, monkeypatch):
    benchmark_root = tmp_path / "benchmarks"
    datasets_dir = benchmark_root / "datasets"
    results_dir = benchmark_root / "results"
    plots_dir = benchmark_root / "plots"
    for path in (datasets_dir, results_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("benchmarks.run_benchmarks.BENCHMARK_ROOT", benchmark_root)
    monkeypatch.setattr("benchmarks.common.BENCHMARK_ROOT", benchmark_root)
    monkeypatch.setattr("benchmarks.visualize_benchmarks.SUMMARY_PATH", results_dir / "summary.csv")
    monkeypatch.setattr("benchmarks.visualize_benchmarks.MEAN_RANK_PATH", results_dir / "mean_rank.csv")
    monkeypatch.setattr("benchmarks.run_benchmarks.SUMMARY_PATH", results_dir / "summary.csv")
    monkeypatch.setattr("benchmarks.run_benchmarks.MEAN_RANK_PATH", results_dir / "mean_rank.csv")
    monkeypatch.setattr("benchmarks.run_benchmarks.required_plot_paths", lambda: [])
    monkeypatch.setattr("benchmarks.run_benchmarks._configure_stdout_utf8", lambda: None)

    monkeypatch.setattr(
        benchmark_models,
        "get_trainable_benchmark_model_ids",
        lambda: ("ctgan", "tvae", "tabddpm"),
    )

    def fake_get_dataset_paths(dataset: str) -> dict[str, Path]:
        return {
            "train": datasets_dir / f"{dataset}_train.csv",
            "test": datasets_dir / f"{dataset}_test.csv",
            "ctgan": benchmark_root / "synthetic" / f"{dataset}_ctgan.csv",
            "tvae": benchmark_root / "synthetic" / f"{dataset}_tvae.csv",
            "tabddpm": benchmark_root / "synthetic" / f"{dataset}_tabddpm.csv",
        }

    monkeypatch.setattr("benchmarks.run_benchmarks.get_dataset_paths", fake_get_dataset_paths)

    for dataset in ("adult", "bank", "covertype", "diabetes"):
        paths = fake_get_dataset_paths(dataset)
        paths["train"].parent.mkdir(parents=True, exist_ok=True)
        paths["train"].write_text("x\n1\n", encoding="utf-8")
        paths["test"].write_text("x\n1\n", encoding="utf-8")
    for dataset in ("bank", "covertype", "diabetes"):
        paths = fake_get_dataset_paths(dataset)
        paths["ctgan"].parent.mkdir(parents=True, exist_ok=True)
        paths["ctgan"].write_text("x\n1\n", encoding="utf-8")
        paths["tvae"].write_text("x\n1\n", encoding="utf-8")

    pd.DataFrame([{"model": "ctgan"}]).to_csv(results_dir / "summary.csv", index=False)
    pd.DataFrame([{"rank": 1}]).to_csv(results_dir / "mean_rank.csv", index=False)

    modules: list[str] = []
    monkeypatch.setattr("benchmarks.run_benchmarks._run_module", lambda module_name: modules.append(module_name))

    main()

    assert "benchmarks.train_benchmark_models" in modules
