from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmarks.export_compute_summary import build_compute_summary, write_compute_summary


def _write_csv(path: Path, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def test_build_compute_summary_tracks_present_and_missing_artifacts(tmp_path):
    benchmark_root = tmp_path / "benchmarks"
    _write_csv(
        benchmark_root / "datasets" / "adult_train.csv",
        pd.DataFrame({"age": [1, 2, 3, 4], "income": [0, 1, 0, 1], "feature": [10, 11, 12, 13]}),
    )
    _write_csv(
        benchmark_root / "datasets" / "adult_test.csv",
        pd.DataFrame({"age": [5, 6], "income": [1, 0], "feature": [14, 15]}),
    )
    _write_csv(
        benchmark_root / "synthetic" / "adult_ctgan.csv",
        pd.DataFrame({"age": [1, 2], "income": [0, 1], "feature": [10, 11]}),
    )
    (benchmark_root / "results").mkdir(parents=True, exist_ok=True)
    (benchmark_root / "results" / "train_benchmark_models_output.log").write_text("adult - CTGAN complete\n", encoding="utf-8")

    summary_df = build_compute_summary(benchmark_root)

    adult_ctgan = summary_df.loc[
        (summary_df["dataset_name"] == "adult") & (summary_df["model_id"] == "ctgan")
    ].iloc[0]
    assert adult_ctgan["row_count"] == 4
    assert adult_ctgan["feature_count"] == 3
    assert bool(adult_ctgan["synthetic_output_present"]) is True
    assert adult_ctgan["synthetic_output_rows"] == 2
    assert bool(adult_ctgan["model_artifact_present"]) is False
    assert "model artifact unavailable" in adult_ctgan["notes"]

    adult_tvae = summary_df.loc[
        (summary_df["dataset_name"] == "adult") & (summary_df["model_id"] == "tvae")
    ].iloc[0]
    assert bool(adult_tvae["synthetic_output_present"]) is False
    assert "synthetic output missing" in adult_tvae["notes"]


def test_write_compute_summary_emits_csv_and_markdown(tmp_path):
    benchmark_root = tmp_path / "benchmarks"
    _write_csv(
        benchmark_root / "datasets" / "adult_train.csv",
        pd.DataFrame({"age": [1, 2], "income": [0, 1], "feature": [10, 11]}),
    )
    _write_csv(
        benchmark_root / "datasets" / "adult_test.csv",
        pd.DataFrame({"age": [3], "income": [1], "feature": [12]}),
    )
    _write_csv(
        benchmark_root / "synthetic" / "adult_ctgan.csv",
        pd.DataFrame({"age": [1], "income": [0], "feature": [10]}),
    )

    outputs = write_compute_summary(benchmark_root)

    csv_text = outputs["csv"].read_text(encoding="utf-8")
    md_text = outputs["markdown"].read_text(encoding="utf-8")
    assert "dataset_name" in csv_text
    assert "adult" in csv_text
    assert "# Compute Summary" in md_text
    assert "Appendix-Ready Inventory" in md_text
