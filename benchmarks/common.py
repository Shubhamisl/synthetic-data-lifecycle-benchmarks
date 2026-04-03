from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"
FAILURE_LOG_PATH = BENCHMARK_ROOT / "results" / "benchmark_failures.log"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    target_col: str
    sensitive_attr: str | None
    domain: str
    fidelity_level: str
    valid_target_values: set[int] | set[str] | None = None


DATASET_REGISTRY = {
    "adult": DatasetSpec(
        name="adult",
        target_col="income",
        sensitive_attr="sex",
        domain="Socioeconomic",
        fidelity_level="Baseline",
        valid_target_values={0, 1},
    ),
    "bank": DatasetSpec(
        name="bank",
        target_col="target",
        sensitive_attr="age_group",
        domain="Marketing / Finance",
        fidelity_level="Level 2",
        valid_target_values={0, 1},
    ),
    "covertype": DatasetSpec(
        name="covertype",
        target_col="Cover_Type",
        sensitive_attr=None,
        domain="Environmental / Ecological",
        fidelity_level="Level 1",
        valid_target_values={1, 2, 3, 4, 5, 6, 7},
    ),
    "diabetes": DatasetSpec(
        name="diabetes",
        target_col="target",
        sensitive_attr="age_group",
        domain="Healthcare",
        fidelity_level="Level 2",
        valid_target_values={0, 1},
    ),
}


def ensure_benchmark_dirs() -> None:
    for name in ("datasets", "synthetic", "results", "plots"):
        (BENCHMARK_ROOT / name).mkdir(parents=True, exist_ok=True)


def get_dataset_paths(name: str) -> dict[str, Path]:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Expected one of {sorted(DATASET_REGISTRY)}")

    return {
        "train": BENCHMARK_ROOT / "datasets" / f"{name}_train.csv",
        "test": BENCHMARK_ROOT / "datasets" / f"{name}_test.csv",
        "ctgan": BENCHMARK_ROOT / "synthetic" / f"{name}_ctgan.csv",
        "tvae": BENCHMARK_ROOT / "synthetic" / f"{name}_tvae.csv",
        "evaluation": BENCHMARK_ROOT / "results" / f"{name}_evaluation.csv",
    }


def validate_dataframe_schema(
    df: pd.DataFrame,
    dataset_name: str,
    required_columns: Iterable[str],
) -> None:
    if df.empty:
        raise ValueError(f"{dataset_name}: dataframe is empty")

    if df.columns.duplicated().any():
        raise ValueError(f"{dataset_name}: duplicate columns detected")

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name}: missing required columns {missing}")

    all_nan_columns = [column for column in df.columns if df[column].isna().all()]
    if all_nan_columns:
        raise ValueError(f"{dataset_name}: all-NaN columns detected {all_nan_columns}")


def validate_required_non_null_columns(
    df: pd.DataFrame,
    dataset_name: str,
    required_columns: Iterable[str],
) -> None:
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"{dataset_name}: missing required column {column!r}")
        if df[column].isna().any():
            raise ValueError(f"{dataset_name}: null values in required column {column!r}")


def validate_target_values(
    df: pd.DataFrame,
    target_col: str,
    valid_values: set[int] | set[str] | None,
    dataset_name: str,
) -> None:
    if target_col not in df.columns:
        raise ValueError(f"{dataset_name}: missing target column {target_col!r}")

    if df[target_col].isna().any():
        raise ValueError(f"{dataset_name}: null values in required column {target_col!r}")

    if valid_values is None:
        return

    observed = set(df[target_col].dropna().unique().tolist())
    invalid_values = observed - valid_values
    if invalid_values:
        raise ValueError(f"{dataset_name}: invalid target values {sorted(invalid_values)}")


def format_summary_table(rows: list[dict[str, object]]) -> str:
    return pd.DataFrame(rows).to_string(index=False)


def log_benchmark_event(stage: str, dataset: str, severity: str, message: str) -> None:
    ensure_benchmark_dirs()
    timestamp = datetime.now(timezone.utc).isoformat()
    with FAILURE_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp}\t{stage}\t{dataset}\t{severity}\t{message}\n")
