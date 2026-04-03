from __future__ import annotations

import pandas as pd
import pytest

from benchmarks.train_benchmark_models import (
    copy_adult_synthetic_artifacts,
    detect_ctgan_discrete_columns,
    _process_training_dataset,
    _print_dataset_completion,
    _coerce_covertype_targets,
    validate_synthetic_dataset,
)


def test_detect_ctgan_discrete_columns_uses_indicator_columns_and_target_for_covertype():
    df = pd.DataFrame(
        {
            "Elevation": [100, 200],
            "Wilderness_Area1": [1, 0],
            "Soil_Type1": [0, 1],
            "Cover_Type": [1, 2],
        }
    )

    assert detect_ctgan_discrete_columns("covertype", df) == [
        "Wilderness_Area1",
        "Soil_Type1",
        "Cover_Type",
    ]


def test_validate_synthetic_dataset_rejects_invalid_target_values():
    train_df = pd.DataFrame({"target": [0, 1], "feature": [1.0, 2.0]})
    synth_df = pd.DataFrame({"target": [0, 2], "feature": [1.0, 2.0]})

    with pytest.raises(ValueError, match="invalid target"):
        validate_synthetic_dataset("bank", train_df, synth_df, "target", {0, 1})


def test_validate_synthetic_dataset_returns_warning_for_constant_columns():
    train_df = pd.DataFrame({"target": [0, 1], "feature": [1.0, 2.0]})
    synth_df = pd.DataFrame({"target": [0, 1], "feature": [9.0, 9.0]})

    warnings = validate_synthetic_dataset("bank", train_df, synth_df, "target", {0, 1})

    assert any("constant" in message for message in warnings)


def test_validate_synthetic_dataset_rejects_non_integer_covertype_targets():
    train_df = pd.DataFrame({"feature": [10, 20], "Cover_Type": [1, 2]})
    synth_df = pd.DataFrame({"feature": [30, 40], "Cover_Type": [1.5, 2.0]})

    with pytest.raises(ValueError, match="integer"):
        validate_synthetic_dataset("covertype", train_df, synth_df, "Cover_Type", {1, 2, 3, 4, 5, 6, 7})


def test_copy_adult_synthetic_artifacts_copies_expected_files(tmp_path, monkeypatch):
    source_ctgan = tmp_path / "ctgan_source.csv"
    source_tvae = tmp_path / "tvae_source.csv"
    benchmark_root = tmp_path / "benchmarks"
    (benchmark_root / "datasets").mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"income": [0, 1], "sex": ["Female", "Male"]}).to_csv(source_ctgan, index=False)
    pd.DataFrame({"income": [1, 0], "sex": ["Male", "Female"]}).to_csv(source_tvae, index=False)
    pd.DataFrame({"income": [0, 1], "sex": ["Female", "Male"]}).to_csv(
        benchmark_root / "datasets" / "adult_train.csv",
        index=False,
    )

    monkeypatch.setattr("benchmarks.common.BENCHMARK_ROOT", benchmark_root)
    monkeypatch.setattr("benchmarks.train_benchmark_models.BENCHMARK_ROOT", benchmark_root)
    monkeypatch.setattr("benchmarks.train_benchmark_models.ADULT_CTGAN_SOURCE", source_ctgan)
    monkeypatch.setattr("benchmarks.train_benchmark_models.ADULT_TVAE_SOURCE", source_tvae)

    ctgan_df, tvae_df = copy_adult_synthetic_artifacts()

    assert (benchmark_root / "synthetic" / "adult_ctgan.csv").exists()
    assert (benchmark_root / "synthetic" / "adult_tvae.csv").exists()
    assert ctgan_df.equals(pd.read_csv(benchmark_root / "synthetic" / "adult_ctgan.csv"))
    assert tvae_df.equals(pd.read_csv(benchmark_root / "synthetic" / "adult_tvae.csv"))


def test_print_dataset_completion_matches_required_output(capsys):
    ctgan_df = pd.DataFrame({"target": [0, 1], "feature": [1.0, 2.0]})
    tvae_df = pd.DataFrame({"target": [1, 0], "feature": [3.0, 4.0]})

    _print_dataset_completion("bank", "target", ctgan_df, tvae_df)

    assert capsys.readouterr().out.splitlines() == [
        "bank - CTGAN complete ✓  |  TVAE complete ✓",
        "Synthetic shape: (2, 2)",
        "Target distribution: {0: 1, 1: 1}",
    ]


def test_coerce_covertype_targets_converts_string_and_float_like_labels():
    synth_df = pd.DataFrame({"feature": [10, 20, 30], "Cover_Type": ["1", 2.0, "3.0"]})

    coerced = _coerce_covertype_targets("covertype", synth_df, "Cover_Type")

    assert coerced["Cover_Type"].tolist() == [1, 2, 3]


def test_process_training_dataset_raises_and_does_not_leave_invalid_artifacts(tmp_path, monkeypatch):
    benchmark_root = tmp_path / "benchmarks"
    (benchmark_root / "synthetic").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("benchmarks.common.BENCHMARK_ROOT", benchmark_root)
    monkeypatch.setattr("benchmarks.train_benchmark_models.BENCHMARK_ROOT", benchmark_root)

    train_df = pd.DataFrame({"target": [0, 1], "feature": [1.0, 2.0]})

    def bad_ctgan_sampler(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"target": [0, 2], "feature": [1.0, 2.0]})

    def unused_tvae_sampler(df: pd.DataFrame) -> pd.DataFrame:
        raise AssertionError("TVAE sampler should not run after CTGAN hard failure")

    with pytest.raises(ValueError, match="invalid target"):
        _process_training_dataset(
            "bank",
            train_df=train_df,
            target_col="target",
            valid_targets={0, 1},
            ctgan_sampler=bad_ctgan_sampler,
            tvae_sampler=unused_tvae_sampler,
        )

    assert not (benchmark_root / "synthetic" / "bank_ctgan.csv").exists()
    assert not (benchmark_root / "synthetic" / "bank_tvae.csv").exists()


def test_process_training_dataset_rejects_nan_training_inputs_before_sampling(tmp_path, monkeypatch):
    benchmark_root = tmp_path / "benchmarks"
    (benchmark_root / "synthetic").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("benchmarks.common.BENCHMARK_ROOT", benchmark_root)
    monkeypatch.setattr("benchmarks.train_benchmark_models.BENCHMARK_ROOT", benchmark_root)

    train_df = pd.DataFrame({"target": [0, 1], "feature": [1.0, None]})

    def should_not_run_ctgan(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
        raise AssertionError("CTGAN sampler should not run when training input has NaNs")

    def should_not_run_tvae(df: pd.DataFrame) -> pd.DataFrame:
        raise AssertionError("TVAE sampler should not run when training input has NaNs")

    with pytest.raises(ValueError, match="training input has NaN values"):
        _process_training_dataset(
            "bank",
            train_df=train_df,
            target_col="target",
            valid_targets={0, 1},
            ctgan_sampler=should_not_run_ctgan,
            tvae_sampler=should_not_run_tvae,
        )


def test_copy_adult_synthetic_artifacts_does_not_promote_invalid_outputs(tmp_path, monkeypatch):
    source_ctgan = tmp_path / "ctgan_source.csv"
    source_tvae = tmp_path / "tvae_source.csv"
    benchmark_root = tmp_path / "benchmarks"
    (benchmark_root / "datasets").mkdir(parents=True, exist_ok=True)
    (benchmark_root / "synthetic").mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"income": [0, 2], "sex": ["Female", "Male"]}).to_csv(source_ctgan, index=False)
    pd.DataFrame({"income": [0, 1], "sex": ["Female", "Male"]}).to_csv(source_tvae, index=False)
    pd.DataFrame({"income": [0, 1], "sex": ["Female", "Male"]}).to_csv(
        benchmark_root / "datasets" / "adult_train.csv",
        index=False,
    )

    monkeypatch.setattr("benchmarks.common.BENCHMARK_ROOT", benchmark_root)
    monkeypatch.setattr("benchmarks.train_benchmark_models.BENCHMARK_ROOT", benchmark_root)
    monkeypatch.setattr("benchmarks.train_benchmark_models.ADULT_CTGAN_SOURCE", source_ctgan)
    monkeypatch.setattr("benchmarks.train_benchmark_models.ADULT_TVAE_SOURCE", source_tvae)

    with pytest.raises(ValueError, match="invalid target"):
        copy_adult_synthetic_artifacts()

    assert not (benchmark_root / "synthetic" / "adult_ctgan.csv").exists()
    assert not (benchmark_root / "synthetic" / "adult_tvae.csv").exists()
