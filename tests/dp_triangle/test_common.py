from __future__ import annotations

from pathlib import Path

import config


def test_triangle_dataset_spec_exposes_adult_metadata():
    from dp_triangle.common import get_dataset_spec

    spec = get_dataset_spec("adult")

    assert spec.name == "adult"
    assert spec.target_col == "income"
    assert spec.sensitive_attr == "sex"
    assert spec.positive_class == 1
    assert spec.supports_full_triangle is True
    assert spec.sensitive_groups == ("Male", "Female")
    assert spec.sensitive_subgroup_value == "Female"


def test_triangle_dataset_spec_marks_covertype_as_partial_triangle():
    from dp_triangle.common import get_dataset_spec

    spec = get_dataset_spec("covertype")

    assert spec.sensitive_attr is None
    assert spec.supports_full_triangle is False
    assert spec.supports_privacy_utility_only is True


def test_triangle_dataset_spec_marks_bank_and_diabetes_with_age_groups():
    from dp_triangle.common import get_dataset_spec

    bank = get_dataset_spec("bank")
    diabetes = get_dataset_spec("diabetes")

    assert bank.sensitive_attr == "age_group"
    assert bank.sensitive_groups == ("young", "older")
    assert bank.sensitive_subgroup_value == "young"
    assert diabetes.sensitive_attr == "age_group"
    assert diabetes.sensitive_groups == ("young", "older")
    assert diabetes.sensitive_subgroup_value == "young"


def test_dataset_input_paths_use_base_data_for_adult():
    from dp_triangle.common import dataset_input_paths

    paths = dataset_input_paths("adult")

    assert paths["train"] == config.DATA_DIR / "adult_train.csv"
    assert paths["test"] == config.DATA_DIR / "adult_test.csv"


def test_dataset_input_paths_use_benchmarks_for_supporting_datasets():
    from dp_triangle.common import dataset_input_paths

    paths = dataset_input_paths("bank")

    expected_root = config.PROJECT_ROOT / "benchmarks" / "datasets"
    assert paths["train"] == expected_root / "bank_train.csv"
    assert paths["test"] == expected_root / "bank_test.csv"


def test_dataset_result_dir_is_dataset_scoped():
    from dp_triangle.common import dataset_result_dir

    adult_dir = dataset_result_dir("adult")
    bank_dir = dataset_result_dir("bank")

    assert adult_dir == config.RESULTS_DIR
    assert bank_dir == config.PROJECT_ROOT / "benchmarks" / "results" / "dp_triangle" / "bank"
    assert isinstance(bank_dir, Path)
