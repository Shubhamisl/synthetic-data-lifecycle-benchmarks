from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmarks.export_reproducibility import (
    build_artifact_inventory,
    build_dataset_manifest,
    build_model_manifest,
    build_reproducibility_manifest,
    build_runtime_manifest,
    write_reproducibility_exports,
)


def _write_csv(path: Path, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def test_build_reproducibility_manifests_cover_files_and_environment(tmp_path):
    benchmark_root = tmp_path / "benchmarks"
    _write_csv(
        benchmark_root / "datasets" / "adult_train.csv",
        pd.DataFrame({"age": [1, 2, 3], "income": [0, 1, 0], "feature": [10, 11, 12]}),
    )
    _write_csv(
        benchmark_root / "datasets" / "adult_test.csv",
        pd.DataFrame({"age": [4], "income": [1], "feature": [13]}),
    )
    _write_csv(
        benchmark_root / "synthetic" / "adult_ctgan.csv",
        pd.DataFrame({"age": [1], "income": [0], "feature": [10]}),
    )
    _write_csv(
        benchmark_root / "results" / "cross_domain_summary.csv",
        pd.DataFrame({"Dataset": ["adult"], "Model": ["CTGAN"]}),
    )
    (benchmark_root / "results").mkdir(parents=True, exist_ok=True)
    (benchmark_root / "results" / "benchmark_run_notes.md").write_text("notes", encoding="utf-8")

    dataset_df = build_dataset_manifest(benchmark_root)
    model_df = build_model_manifest(benchmark_root)
    inventory_df = build_artifact_inventory(benchmark_root)
    runtime_df = build_runtime_manifest(benchmark_root)
    manifest_df = build_reproducibility_manifest(benchmark_root)

    adult_row = dataset_df.loc[dataset_df["dataset_name"] == "adult"].iloc[0]
    assert adult_row["train_rows"] == 3
    assert adult_row["train_features"] == 3
    assert bool(adult_row["train_present"]) is True

    ctgan_row = model_df.loc[model_df["model_id"] == "ctgan"].iloc[0]
    assert ctgan_row["display_name"] == "CTGAN"

    assert "cross-domain summary" in inventory_df["name"].tolist()
    assert "python_version" in runtime_df["key"].tolist()
    assert "section" in manifest_df.columns
    assert set(manifest_df["section"]).issuperset({"dataset", "model", "runtime", "artifact", "context"})


def test_write_reproducibility_exports_emits_csv_and_markdown(tmp_path):
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

    outputs = write_reproducibility_exports(benchmark_root)

    manifest_text = outputs["reproducibility_manifest"].read_text(encoding="utf-8")
    markdown_text = outputs["environment_summary"].read_text(encoding="utf-8")
    assert "section" in manifest_text
    assert "dataset" in manifest_text
    assert "# Environment Summary" in markdown_text
    assert "Artifact Inventory" in markdown_text
