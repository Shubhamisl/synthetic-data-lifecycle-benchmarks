"""Shared dataset metadata and path helpers for Direction 3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import config


@dataclass(frozen=True)
class TriangleDatasetSpec:
    """
    Describe dataset-specific metadata for Direction 3 analyses.

    Inputs: dataset configuration constants.
    Outputs: immutable dataset metadata record.
    Lifecycle stage: Shared metadata for Stages 1-7.
    Reference: Direction 3 dataset generalization design for Adult, Bank, Diabetes, and Covertype.
    """

    name: str
    target_col: str
    sensitive_attr: str | None
    positive_class: int | str
    supports_full_triangle: bool
    supports_privacy_utility_only: bool
    sensitive_groups: tuple[str | int, str | int] | None = None
    sensitive_subgroup_value: str | int | None = None
    subgroup_positive_label: str | int | None = None
    subgroup_sensitive_label: str | int | None = None


DATASET_SPECS: dict[str, TriangleDatasetSpec] = {
    "adult": TriangleDatasetSpec(
        name="adult",
        target_col=config.DP_TARGET_COL,
        sensitive_attr=config.DP_SENSITIVE_COL,
        positive_class=config.DP_MINORITY_CLASS,
        supports_full_triangle=True,
        supports_privacy_utility_only=False,
        sensitive_groups=("Male", "Female"),
        sensitive_subgroup_value="Female",
        subgroup_positive_label=config.DP_MINORITY_CLASS,
        subgroup_sensitive_label="Female",
    ),
    "bank": TriangleDatasetSpec(
        name="bank",
        target_col="target",
        sensitive_attr="age_group",
        positive_class=1,
        supports_full_triangle=True,
        supports_privacy_utility_only=False,
        sensitive_groups=("young", "older"),
        sensitive_subgroup_value="young",
        subgroup_positive_label=1,
        subgroup_sensitive_label="young",
    ),
    "diabetes": TriangleDatasetSpec(
        name="diabetes",
        target_col="target",
        sensitive_attr="age_group",
        positive_class=1,
        supports_full_triangle=True,
        supports_privacy_utility_only=False,
        sensitive_groups=("young", "older"),
        sensitive_subgroup_value="young",
        subgroup_positive_label=1,
        subgroup_sensitive_label="young",
    ),
    "covertype": TriangleDatasetSpec(
        name="covertype",
        target_col="Cover_Type",
        sensitive_attr=None,
        positive_class=1,
        supports_full_triangle=False,
        supports_privacy_utility_only=True,
    ),
}


def get_dataset_spec(name: str) -> TriangleDatasetSpec:
    """
    Return Direction 3 metadata for a named dataset.

    Inputs: dataset name.
    Outputs: triangle dataset specification.
    Lifecycle stage: Shared metadata for Stages 1-7.
    Reference: Direction 3 multi-dataset extension design.
    """
    try:
        return DATASET_SPECS[name]
    except KeyError as exc:
        expected = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unknown Direction 3 dataset {name!r}. Expected one of: {expected}") from exc


def dataset_input_paths(name: str) -> dict[str, Path]:
    """
    Return canonical train/test paths for a Direction 3 dataset.

    Inputs: dataset name.
    Outputs: mapping containing train and test CSV paths.
    Lifecycle stage: Shared I/O for Stages 0-4.
    Reference: project path portability design for Adult and benchmark datasets.
    """
    spec = get_dataset_spec(name)
    if spec.name == "adult":
        return {
            "train": config.DATA_DIR / "adult_train.csv",
            "test": config.DATA_DIR / "adult_test.csv",
        }

    benchmark_dir = config.PROJECT_ROOT / "benchmarks" / "datasets"
    return {
        "train": benchmark_dir / f"{spec.name}_train.csv",
        "test": benchmark_dir / f"{spec.name}_test.csv",
    }


def dataset_result_dir(name: str) -> Path:
    """
    Return the result directory for a Direction 3 dataset.

    Inputs: dataset name.
    Outputs: dataset-specific result directory path.
    Lifecycle stage: Shared I/O for Stages 4-7.
    Reference: Direction 3 output organization for flagship and supporting datasets.
    """
    spec = get_dataset_spec(name)
    if spec.name == "adult":
        return config.RESULTS_DIR
    return config.PROJECT_ROOT / "benchmarks" / "results" / "dp_triangle" / spec.name
