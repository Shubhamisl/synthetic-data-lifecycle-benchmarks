from __future__ import annotations

import pandas as pd
import pytest

from benchmarks.common import (
    DATASET_REGISTRY,
    DatasetSpec,
    ensure_benchmark_dirs,
    format_summary_table,
    get_dataset_paths,
    validate_dataframe_schema,
    validate_target_values,
)


def test_dataset_registry_contains_expected_benchmarks():
    assert set(DATASET_REGISTRY) == {"adult", "bank", "covertype", "diabetes"}
    assert isinstance(DATASET_REGISTRY["adult"], DatasetSpec)
    assert DATASET_REGISTRY["adult"].sensitive_attr == "sex"
    assert DATASET_REGISTRY["bank"].sensitive_attr == "age_group"
    assert DATASET_REGISTRY["bank"].valid_target_values == {0, 1}
    assert DATASET_REGISTRY["covertype"].valid_target_values == {1, 2, 3, 4, 5, 6, 7}
    assert DATASET_REGISTRY["diabetes"].sensitive_attr == "age_group"


def test_ensure_benchmark_dirs_creates_expected_folders(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmarks.common.BENCHMARK_ROOT", tmp_path)

    ensure_benchmark_dirs()

    for name in ["datasets", "synthetic", "results", "plots"]:
        assert (tmp_path / name).is_dir()


def test_get_dataset_paths_uses_benchmark_root(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmarks.common.BENCHMARK_ROOT", tmp_path)

    paths = get_dataset_paths("adult")

    assert paths["train"] == tmp_path / "datasets" / "adult_train.csv"
    assert paths["test"] == tmp_path / "datasets" / "adult_test.csv"
    assert paths["ctgan"] == tmp_path / "synthetic" / "adult_ctgan.csv"
    assert paths["tvae"] == tmp_path / "synthetic" / "adult_tvae.csv"
    assert paths["evaluation"] == tmp_path / "results" / "adult_evaluation.csv"


def test_get_dataset_paths_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_dataset_paths("unknown")


def test_validate_target_values_rejects_invalid_class():
    df = pd.DataFrame({"target": [0, 1, 3]})

    with pytest.raises(ValueError, match="invalid target"):
        validate_target_values(df, "target", {0, 1}, "bank")


def test_validate_target_values_rejects_missing_target_column():
    df = pd.DataFrame({"feature": [1, 2, 3]})

    with pytest.raises(ValueError, match="missing target column"):
        validate_target_values(df, "target", {0, 1}, "bank")


def test_validate_dataframe_schema_rejects_duplicate_columns():
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])

    with pytest.raises(ValueError, match="duplicate"):
        validate_dataframe_schema(df, "adult", required_columns=["a"])


def test_format_summary_table_includes_headers_and_values():
    rows = [
        {
            "Dataset": "adult",
            "Samples": 45222,
            "Features": 15,
            "Target classes": 2,
            "Sensitive attr": "sex",
        }
    ]

    table = format_summary_table(rows)

    assert "Dataset" in table
    assert "adult" in table
    assert "Sensitive attr" in table
