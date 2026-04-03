from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import pytest

from benchmarks.download_datasets import (
    load_and_validate_adult_dataset_pair,
    load_bank_dataframe_with_fallback,
    preprocess_bank,
    preprocess_diabetes,
    split_stratified,
    stratified_sample,
    summarize_dataset,
    validate_saved_dataset_pair,
)


def test_preprocess_bank_creates_age_group_drops_leaky_columns_and_encodes_target():
    df = pd.DataFrame(
        {
            "age": [25, 55],
            "duration": [10, 20],
            "job": ["admin.", "services"],
            "y": ["yes", "no"],
        }
    )

    out = preprocess_bank(df)

    assert "age_group" in out.columns
    assert "age" not in out.columns
    assert "duration" not in out.columns
    assert out["age_group"].tolist() == ["young", "older"]
    assert out["target"].tolist() == [1, 0]


def test_preprocess_bank_fills_missing_categorical_values_with_unknown():
    df = pd.DataFrame(
        {
            "age": [25, 55],
            "duration": [10, 20],
            "job": ["admin.", None],
            "education": [None, "secondary"],
            "contact": [None, "cellular"],
            "poutcome": ["success", None],
            "y": ["yes", "no"],
        }
    )

    out = preprocess_bank(df)

    assert out[["job", "education", "contact", "poutcome"]].isna().sum().sum() == 0
    assert out.loc[1, "job"] == "unknown"
    assert out.loc[0, "education"] == "unknown"
    assert out.loc[0, "contact"] == "unknown"
    assert out.loc[1, "poutcome"] == "unknown"


def test_preprocess_diabetes_replaces_zero_missing_values_and_drops_age():
    df = pd.DataFrame(
        {
            "pregnancies": [1, 2, 3],
            "glucose": [0, 100, 120],
            "blood_pressure": [70, 0, 80],
            "skin_thickness": [0, 20, 25],
            "insulin": [0, 80, 120],
            "bmi": [0.0, 30.0, 32.0],
            "diabetes_pedigree": [0.2, 0.4, 0.5],
            "age": [30, 50, 40],
            "target": [0, 1, 0],
        }
    )

    out = preprocess_diabetes(df)

    assert (
        out[["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]] == 0
    ).sum().sum() == 0
    assert "age_group" in out.columns
    assert "age" not in out.columns


def test_split_stratified_preserves_both_target_classes_in_train_and_test():
    df = pd.DataFrame({"feature": range(10), "target": [0, 1] * 5})

    train_df, test_df = split_stratified(df, "target", test_size=0.2, random_state=42)

    assert set(train_df["target"]) == {0, 1}
    assert set(test_df["target"]) == {0, 1}


def test_summarize_dataset_returns_expected_keys_and_values():
    df = pd.DataFrame({"x": [1, 2], "target": [0, 1], "age_group": ["young", "older"]})

    summary = summarize_dataset("bank", df, "target", "age_group")

    assert summary == {
        "Dataset": "bank",
        "Samples": 2,
        "Features": 3,
        "Target classes": 2,
        "Sensitive attr": "age_group",
    }


def test_validate_saved_dataset_pair_rejects_mismatched_columns():
    train_df = pd.DataFrame({"feature": [1, 2], "target": [0, 1], "age_group": ["young", "older"]})
    test_df = pd.DataFrame({"other": [3, 4], "target": [0, 1], "age_group": ["young", "older"]})

    with pytest.raises(ValueError, match="columns do not match"):
        validate_saved_dataset_pair(
            "bank",
            train_df,
            test_df,
            target_col="target",
            sensitive_attr="age_group",
            valid_target_values={0, 1},
        )


def test_preprocess_bank_rejects_unexpected_source_labels():
    df = pd.DataFrame(
        {
            "age": [25, 55],
            "duration": [10, 20],
            "job": ["admin.", "services"],
            "y": ["yes", "maybe"],
        }
    )

    with pytest.raises(ValueError, match="unexpected target labels"):
        preprocess_bank(df)


def test_validate_saved_dataset_pair_requires_sensitive_attribute_in_test_split():
    train_df = pd.DataFrame({"feature": [1, 2], "target": [0, 1], "age_group": ["young", "older"]})
    test_df = pd.DataFrame({"feature": [3, 4], "target": [0, 1]})

    with pytest.raises(ValueError, match="missing sensitive attribute"):
        validate_saved_dataset_pair(
            "bank",
            train_df,
            test_df,
            target_col="target",
            sensitive_attr="age_group",
            valid_target_values={0, 1},
        )


def test_validate_saved_dataset_pair_rejects_null_target_values():
    train_df = pd.DataFrame({"feature": [1, 2], "target": [0, None], "age_group": ["young", "older"]})
    test_df = pd.DataFrame({"feature": [3, 4], "target": [0, 1], "age_group": ["young", "older"]})

    with pytest.raises(ValueError, match="null values in required column 'target'"):
        validate_saved_dataset_pair(
            "bank",
            train_df,
            test_df,
            target_col="target",
            sensitive_attr="age_group",
            valid_target_values={0, 1},
        )


def test_validate_saved_dataset_pair_rejects_null_sensitive_values():
    train_df = pd.DataFrame({"feature": [1, 2], "target": [0, 1], "age_group": ["young", None]})
    test_df = pd.DataFrame({"feature": [3, 4], "target": [0, 1], "age_group": ["young", "older"]})

    with pytest.raises(ValueError, match="null values in required column 'age_group'"):
        validate_saved_dataset_pair(
            "bank",
            train_df,
            test_df,
            target_col="target",
            sensitive_attr="age_group",
            valid_target_values={0, 1},
        )


def test_load_and_validate_adult_dataset_pair_rejects_missing_sensitive_attribute(tmp_path, monkeypatch):
    source_train = tmp_path / "adult_train.csv"
    source_test = tmp_path / "adult_test.csv"
    evaluation = tmp_path / "adult_eval.csv"
    benchmark_root = tmp_path / "benchmarks"

    pd.DataFrame({"age": [20, 30], "income": [0, 1]}).to_csv(source_train, index=False)
    pd.DataFrame({"age": [40, 50], "income": [0, 1]}).to_csv(source_test, index=False)
    pd.DataFrame(
        {
            "Model": ["Real", "CTGAN", "TVAE"],
            "JS_Divergence": [0.0, 0.1, 0.2],
            "TSTR_Accuracy": [85.0, 80.0, 79.0],
            "MIA_Advantage": [0.0, 0.1, 0.1],
            "Demographic_Parity": [0.2, 0.2, 0.2],
        }
    ).to_csv(evaluation, index=False)

    monkeypatch.setattr("benchmarks.download_datasets.ADULT_TRAIN_SOURCE", source_train)
    monkeypatch.setattr("benchmarks.download_datasets.ADULT_TEST_SOURCE", source_test)
    monkeypatch.setattr("benchmarks.download_datasets.ADULT_EVALUATION_SOURCE", evaluation)
    monkeypatch.setattr("benchmarks.common.BENCHMARK_ROOT", benchmark_root)
    monkeypatch.setattr("benchmarks.download_datasets.BENCHMARK_ROOT", benchmark_root)

    with pytest.raises(ValueError, match="missing sensitive attribute"):
        load_and_validate_adult_dataset_pair()


def test_load_and_validate_adult_dataset_pair_rejects_invalid_evaluation_csv(tmp_path, monkeypatch):
    source_train = tmp_path / "adult_train.csv"
    source_test = tmp_path / "adult_test.csv"
    evaluation = tmp_path / "adult_eval.csv"
    benchmark_root = tmp_path / "benchmarks"

    pd.DataFrame({"age": [20, 30], "income": [0, 1], "sex": ["Female", "Male"]}).to_csv(source_train, index=False)
    pd.DataFrame({"age": [40, 50], "income": [0, 1], "sex": ["Female", "Male"]}).to_csv(source_test, index=False)
    pd.DataFrame({"Model": ["Real"], "TSTR_Accuracy": [85.0]}).to_csv(evaluation, index=False)

    monkeypatch.setattr("benchmarks.download_datasets.ADULT_TRAIN_SOURCE", source_train)
    monkeypatch.setattr("benchmarks.download_datasets.ADULT_TEST_SOURCE", source_test)
    monkeypatch.setattr("benchmarks.download_datasets.ADULT_EVALUATION_SOURCE", evaluation)
    monkeypatch.setattr("benchmarks.common.BENCHMARK_ROOT", benchmark_root)
    monkeypatch.setattr("benchmarks.download_datasets.BENCHMARK_ROOT", benchmark_root)

    with pytest.raises(ValueError, match="adult evaluation"):
        load_and_validate_adult_dataset_pair()


def test_load_and_validate_adult_dataset_pair_rejects_null_required_synthetic_metrics(tmp_path, monkeypatch):
    source_train = tmp_path / "adult_train.csv"
    source_test = tmp_path / "adult_test.csv"
    evaluation = tmp_path / "adult_eval.csv"
    benchmark_root = tmp_path / "benchmarks"

    pd.DataFrame({"age": [20, 30], "income": [0, 1], "sex": ["Female", "Male"]}).to_csv(source_train, index=False)
    pd.DataFrame({"age": [40, 50], "income": [0, 1], "sex": ["Female", "Male"]}).to_csv(source_test, index=False)
    pd.DataFrame(
        {
            "Model": ["Real", "CTGAN", "TVAE"],
            "JS_Divergence": [None, None, 0.2],
            "TSTR_Accuracy": [85.0, 80.0, 79.0],
            "MIA_Advantage": [None, 0.1, 0.1],
            "Demographic_Parity": [0.2, None, 0.2],
        }
    ).to_csv(evaluation, index=False)

    monkeypatch.setattr("benchmarks.download_datasets.ADULT_TRAIN_SOURCE", source_train)
    monkeypatch.setattr("benchmarks.download_datasets.ADULT_TEST_SOURCE", source_test)
    monkeypatch.setattr("benchmarks.download_datasets.ADULT_EVALUATION_SOURCE", evaluation)
    monkeypatch.setattr("benchmarks.common.BENCHMARK_ROOT", benchmark_root)
    monkeypatch.setattr("benchmarks.download_datasets.BENCHMARK_ROOT", benchmark_root)

    with pytest.raises(ValueError, match="null values in required synthetic metric"):
        load_and_validate_adult_dataset_pair()


def test_load_bank_dataframe_with_fallback_uses_direct_zip_when_ucimlrepo_fails(monkeypatch):
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("bank-full.csv", 'age;"job";"y"\n30;"admin.";"yes"\n')

    class _Response:
        def __init__(self, payload: bytes):
            self.payload = payload

        def read(self) -> bytes:
            return self.payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("benchmarks.download_datasets.fetch_ucirepo", lambda id: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr("benchmarks.download_datasets.urlopen", lambda url: _Response(buffer.getvalue()))

    df = load_bank_dataframe_with_fallback()

    assert df.to_dict(orient="records") == [{"age": 30, "job": "admin.", "y": "yes"}]


def test_load_bank_dataframe_with_fallback_rejects_malformed_zip_payload(monkeypatch):
    class _Response:
        def read(self) -> bytes:
            return b"not-a-zip"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("benchmarks.download_datasets.fetch_ucirepo", lambda id: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr("benchmarks.download_datasets.urlopen", lambda url: _Response())

    with pytest.raises(RuntimeError, match="unable to fetch dataset"):
        load_bank_dataframe_with_fallback()


def test_stratified_sample_falls_back_to_five_percent_only_for_legitimate_sample_size_failures(monkeypatch):
    df = pd.DataFrame({"feature": range(40), "Cover_Type": [1] * 38 + [2] * 2})
    call_sizes: list[float] = []
    real_split = __import__("benchmarks.download_datasets", fromlist=["train_test_split"]).train_test_split

    def fake_split(*args, **kwargs):
        call_sizes.append(kwargs["train_size"])
        if kwargs["train_size"] == 0.10:
            raise ValueError("The least populated class in y has only 1 member, which is too few.")
        return real_split(*args, **kwargs)

    monkeypatch.setattr("benchmarks.download_datasets.train_test_split", fake_split)

    sample_df = stratified_sample(df, "Cover_Type", sample_fraction=0.10, fallback_fraction=0.05, random_state=42)

    assert call_sizes == [0.10, 0.05]
    assert not sample_df.empty


def test_stratified_sample_does_not_fall_back_on_arbitrary_exceptions(monkeypatch):
    df = pd.DataFrame({"feature": range(20), "Cover_Type": [1] * 10 + [2] * 10})

    def fake_split(*args, **kwargs):
        raise TypeError("corrupted input")

    monkeypatch.setattr("benchmarks.download_datasets.train_test_split", fake_split)

    with pytest.raises(TypeError, match="corrupted input"):
        stratified_sample(df, "Cover_Type", sample_fraction=0.10, fallback_fraction=0.05, random_state=42)


def test_stratified_sample_does_not_fall_back_on_unrelated_valueerror(monkeypatch):
    df = pd.DataFrame({"feature": range(20), "Cover_Type": [1] * 10 + [2] * 10})
    call_sizes: list[float] = []

    def fake_split(*args, **kwargs):
        call_sizes.append(kwargs["train_size"])
        raise ValueError("totally unrelated parsing failure")

    monkeypatch.setattr("benchmarks.download_datasets.train_test_split", fake_split)

    with pytest.raises(ValueError, match="totally unrelated parsing failure"):
        stratified_sample(df, "Cover_Type", sample_fraction=0.10, fallback_fraction=0.05, random_state=42)

    assert call_sizes == [0.10]
