# Cross-Domain Benchmarking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new `benchmarks/` package that downloads or copies four datasets, trains or copies synthetic outputs, evaluates cross-domain quality across four metrics, generates four plots, and runs the four required commands in sequence without modifying protected Adult lifecycle artifacts.

**Architecture:** Add a script-first benchmark package under `benchmarks/` with a small `common.py` helper for dataset metadata, validation, logging, encoding, and reusable formatting. Keep each stage independently runnable, fail fast on corrupted or inconsistent inputs, and mirror the existing lifecycle style while adding benchmark-specific tests under a new `tests/benchmarks/` tree.

**Tech Stack:** Python, pandas, numpy, scikit-learn, ctgan, sdv, matplotlib, seaborn, pytest, ucimlrepo

---

## File Structure Map

### Create

- `benchmarks/__init__.py`
- `benchmarks/common.py`
- `benchmarks/download_datasets.py`
- `benchmarks/train_benchmark_models.py`
- `benchmarks/evaluate_benchmarks.py`
- `benchmarks/visualize_benchmarks.py`
- `benchmarks/run_benchmarks.py`
- `tests/benchmarks/__init__.py`
- `tests/benchmarks/test_common.py`
- `tests/benchmarks/test_download_datasets.py`
- `tests/benchmarks/test_train_benchmark_models.py`
- `tests/benchmarks/test_evaluate_benchmarks.py`
- `tests/benchmarks/test_visualize_benchmarks.py`

### Runtime Directories Created By Code

- `benchmarks/datasets/`
- `benchmarks/synthetic/`
- `benchmarks/results/`
- `benchmarks/plots/`

### Modify

- none required outside the new `benchmarks/` and `tests/benchmarks/` paths

## Task 1: Scaffold The Benchmark Package And Shared Helpers

**Files:**
- Create: `benchmarks/__init__.py`
- Create: `benchmarks/common.py`
- Create: `tests/benchmarks/__init__.py`
- Create: `tests/benchmarks/test_common.py`

- [ ] **Step 1: Write the failing shared-helper tests**

```python
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.common import (
    BENCHMARK_ROOT,
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
    assert DATASET_REGISTRY["bank"].sensitive_attr == "age_group"
    assert DATASET_REGISTRY["covertype"].valid_target_values == {1, 2, 3, 4, 5, 6, 7}


def test_ensure_benchmark_dirs_creates_expected_folders(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmarks.common.BENCHMARK_ROOT", tmp_path)
    ensure_benchmark_dirs()
    for name in ["datasets", "synthetic", "results", "plots"]:
        assert (tmp_path / name).exists()


def test_validate_target_values_rejects_invalid_class():
    df = pd.DataFrame({"target": [0, 1, 3]})
    with pytest.raises(ValueError, match="invalid target"):
        validate_target_values(df, "target", {0, 1}, "bank")


def test_validate_dataframe_schema_rejects_duplicate_columns():
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(ValueError, match="duplicate"):
        validate_dataframe_schema(df, "adult", required_columns=["a"])


def test_format_summary_table_contains_header():
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/benchmarks/test_common.py -q`
Expected: FAIL with import errors because `benchmarks.common` and helper symbols do not exist yet.

- [ ] **Step 3: Write the minimal shared helper implementation**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_ROOT = PROJECT_ROOT / "benchmarks"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    target_col: str
    sensitive_attr: str | None
    domain: str
    fidelity_level: str
    valid_target_values: set | None = None


DATASET_REGISTRY = {
    "adult": DatasetSpec("adult", "income", "sex", "Socioeconomic", "Baseline"),
    "bank": DatasetSpec("bank", "target", "age_group", "Marketing / Finance", "Level 2", {0, 1}),
    "covertype": DatasetSpec("covertype", "Cover_Type", None, "Environmental / Ecological", "Level 1", {1, 2, 3, 4, 5, 6, 7}),
    "diabetes": DatasetSpec("diabetes", "target", "age_group", "Healthcare", "Level 2", {0, 1}),
}


def ensure_benchmark_dirs() -> None:
    for name in ["datasets", "synthetic", "results", "plots"]:
        (BENCHMARK_ROOT / name).mkdir(parents=True, exist_ok=True)


def get_dataset_paths(name: str) -> dict[str, Path]:
    return {
        "train": BENCHMARK_ROOT / "datasets" / f"{name}_train.csv",
        "test": BENCHMARK_ROOT / "datasets" / f"{name}_test.csv",
        "ctgan": BENCHMARK_ROOT / "synthetic" / f"{name}_ctgan.csv",
        "tvae": BENCHMARK_ROOT / "synthetic" / f"{name}_tvae.csv",
        "evaluation": BENCHMARK_ROOT / "results" / f"{name}_evaluation.csv",
    }


def validate_dataframe_schema(df: pd.DataFrame, dataset_name: str, required_columns: list[str]) -> None:
    if df.empty:
        raise ValueError(f"{dataset_name}: empty dataframe")
    if df.columns.duplicated().any():
        raise ValueError(f"{dataset_name}: duplicate columns detected")
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name}: missing required columns {missing}")


def validate_target_values(df: pd.DataFrame, target_col: str, valid_values: set | None, dataset_name: str) -> None:
    if valid_values is None:
        return
    observed = set(pd.Series(df[target_col]).dropna().unique().tolist())
    if not observed.issubset(valid_values):
        raise ValueError(f"{dataset_name}: invalid target values {sorted(observed - valid_values)}")


def format_summary_table(rows: list[dict[str, object]]) -> str:
    return pd.DataFrame(rows).to_string(index=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/benchmarks/test_common.py -q`
Expected: PASS with 5 passed.

- [ ] **Step 5: Commit**

If `git rev-parse --is-inside-work-tree` succeeds:

```bash
git add benchmarks/__init__.py benchmarks/common.py tests/benchmarks/__init__.py tests/benchmarks/test_common.py
git commit -m "feat: scaffold benchmark helpers"
```

If the workspace is still not a git repo, record the skipped commit in working notes and continue.

## Task 2: Implement Dataset Download, Copy, And Preprocessing

**Files:**
- Modify: `benchmarks/common.py`
- Create: `benchmarks/download_datasets.py`
- Create: `tests/benchmarks/test_download_datasets.py`

- [ ] **Step 1: Write the failing dataset-download tests**

```python
from pathlib import Path

import pandas as pd

from benchmarks.download_datasets import (
    preprocess_bank,
    preprocess_diabetes,
    split_stratified,
    summarize_dataset,
)


def test_preprocess_bank_creates_age_group_and_binary_target():
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
    assert out["target"].tolist() == [1, 0]


def test_preprocess_diabetes_replaces_zero_missing_values():
    df = pd.DataFrame(
        {
            "pregnancies": [1, 2],
            "glucose": [0, 100],
            "blood_pressure": [70, 0],
            "skin_thickness": [0, 20],
            "insulin": [0, 80],
            "bmi": [0.0, 30.0],
            "diabetes_pedigree": [0.2, 0.4],
            "age": [30, 50],
            "target": [0, 1],
        }
    )
    out = preprocess_diabetes(df)
    assert (out[["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]] == 0).sum().sum() == 0
    assert "age_group" in out.columns
    assert "age" not in out.columns


def test_split_stratified_preserves_target_classes():
    df = pd.DataFrame({"feature": range(10), "target": [0, 1] * 5})
    train_df, test_df = split_stratified(df, "target", test_size=0.2, random_state=42)
    assert set(train_df["target"]) == {0, 1}
    assert set(test_df["target"]) == {0, 1}


def test_summarize_dataset_reports_expected_keys():
    df = pd.DataFrame({"x": [1, 2], "target": [0, 1], "age_group": ["young", "older"]})
    summary = summarize_dataset("bank", df, "target", "age_group")
    assert summary["Dataset"] == "bank"
    assert summary["Target classes"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/benchmarks/test_download_datasets.py -q`
Expected: FAIL because the preprocessing functions and summary helpers are not implemented yet.

- [ ] **Step 3: Write the minimal download and preprocessing implementation**

```python
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from .common import DATASET_REGISTRY, ensure_benchmark_dirs, format_summary_table, get_dataset_paths


def split_stratified(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(df, test_size=test_size, stratify=df[target_col], random_state=random_state)


def preprocess_bank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns={"y": "target"})
    out["target"] = out["target"].map({"yes": 1, "no": 0})
    out["age_group"] = out["age"].apply(lambda value: "young" if value <= 40 else "older")
    out = out.drop(columns=[col for col in ["age", "duration"] if col in out.columns])
    return out


def preprocess_diabetes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]:
        median_value = out.loc[out[col] != 0, col].median()
        out.loc[out[col] == 0, col] = median_value
    out["age_group"] = out["age"].apply(lambda value: "young" if value <= 35 else "older")
    return out.drop(columns=["age"])


def summarize_dataset(name: str, df: pd.DataFrame, target_col: str, sensitive_attr: str | None) -> dict[str, object]:
    return {
        "Dataset": name,
        "Samples": len(df),
        "Features": len(df.columns),
        "Target classes": df[target_col].nunique(),
        "Sensitive attr": sensitive_attr or "None",
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/benchmarks/test_download_datasets.py -q`
Expected: PASS with 4 passed.

- [ ] **Step 5: Expand implementation to full stage behavior**

```python
def main():
    ensure_benchmark_dirs()
    summary_rows = []
    for dataset_name in ["adult", "bank", "covertype", "diabetes"]:
        spec = DATASET_REGISTRY[dataset_name]
        paths = get_dataset_paths(dataset_name)
        if dataset_name == "adult":
            train_df = pd.read_csv(PROJECT_ROOT / "data" / "adult_train.csv")
            test_df = pd.read_csv(PROJECT_ROOT / "data" / "adult_test.csv")
        elif dataset_name == "bank":
            raw_df = load_bank_dataframe_with_fallback()
            processed_df = preprocess_bank(raw_df)
            train_df, test_df = split_stratified(processed_df, spec.target_col, test_size=0.2, random_state=42)
        elif dataset_name == "covertype":
            raw_df = load_covertype_dataframe_with_fallback()
            sampled_df = stratified_sample(raw_df, "Cover_Type", sample_fraction=0.10, fallback_fraction=0.05, random_state=42)
            train_df, test_df = split_stratified(sampled_df, spec.target_col, test_size=0.2, random_state=42)
        else:
            raw_df = load_diabetes_dataframe()
            processed_df = preprocess_diabetes(raw_df)
            train_df, test_df = split_stratified(processed_df, spec.target_col, test_size=0.2, random_state=42)

        train_df.to_csv(paths["train"], index=False)
        test_df.to_csv(paths["test"], index=False)
        train_df = pd.read_csv(paths["train"])
        test_df = pd.read_csv(paths["test"])
        validate_split_pair(dataset_name, train_df, test_df, spec)
        print_dataset_details(dataset_name, train_df, test_df, spec.target_col)
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        summary_rows.append(summarize_dataset(dataset_name, combined_df, spec.target_col, spec.sensitive_attr))
    print("Summary table")
    print(format_summary_table(summary_rows))
```

Include try/except per dataset, fallback URLs for UCI data, hard-failure logging, and explicit disk reload validation before success is reported.

- [ ] **Step 6: Run focused tests plus the stage**

Run: `python -m pytest tests/benchmarks/test_common.py tests/benchmarks/test_download_datasets.py -q`
Expected: PASS

Run: `python -m benchmarks.download_datasets`
Expected: prints four dataset summaries and the final summary table, then exits 0.

- [ ] **Step 7: Commit**

If the workspace is a git repo:

```bash
git add benchmarks/common.py benchmarks/download_datasets.py tests/benchmarks/test_download_datasets.py
git commit -m "feat: add benchmark dataset preparation"
```

If not, continue without commit.

## Task 3: Implement Benchmark Training And Synthetic Validation

**Files:**
- Modify: `benchmarks/common.py`
- Create: `benchmarks/train_benchmark_models.py`
- Create: `tests/benchmarks/test_train_benchmark_models.py`

- [ ] **Step 1: Write the failing training-validation tests**

```python
import pandas as pd
import pytest

from benchmarks.train_benchmark_models import (
    detect_ctgan_discrete_columns,
    validate_synthetic_dataset,
)


def test_detect_ctgan_discrete_columns_uses_all_columns_for_covertype():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "Cover_Type": [1, 2]})
    assert detect_ctgan_discrete_columns("covertype", df) == ["A", "B", "Cover_Type"]


def test_validate_synthetic_dataset_rejects_invalid_target_values():
    train_df = pd.DataFrame({"target": [0, 1], "feature": [1.0, 2.0]})
    synth_df = pd.DataFrame({"target": [0, 2], "feature": [1.0, 2.0]})
    with pytest.raises(ValueError, match="invalid target"):
        validate_synthetic_dataset("bank", train_df, synth_df, "target", {0, 1})


def test_validate_synthetic_dataset_warns_on_constant_column():
    train_df = pd.DataFrame({"target": [0, 1], "feature": [1.0, 2.0]})
    synth_df = pd.DataFrame({"target": [0, 1], "feature": [9.0, 9.0]})
    warnings = validate_synthetic_dataset("bank", train_df, synth_df, "target", {0, 1})
    assert any("constant" in message for message in warnings)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/benchmarks/test_train_benchmark_models.py -q`
Expected: FAIL because training helper functions do not exist yet.

- [ ] **Step 3: Write minimal training helper implementation**

```python
from __future__ import annotations

import pandas as pd


def detect_ctgan_discrete_columns(dataset_name: str, train_df: pd.DataFrame) -> list[str]:
    if dataset_name == "covertype":
        return train_df.columns.tolist()
    return train_df.select_dtypes(include=["object", "category"]).columns.tolist()


def validate_synthetic_dataset(dataset_name: str, train_df: pd.DataFrame, synth_df: pd.DataFrame, target_col: str, valid_targets: set | None) -> list[str]:
    if list(train_df.columns) != list(synth_df.columns):
        raise ValueError(f"{dataset_name}: schema mismatch")
    if synth_df.isna().sum().sum():
        raise ValueError(f"{dataset_name}: synthetic output has NaN values")
    if valid_targets is not None and not set(synth_df[target_col].unique()).issubset(valid_targets):
        raise ValueError(f"{dataset_name}: invalid target values")
    warnings = []
    for col in synth_df.columns:
        if synth_df[col].nunique(dropna=False) == 1:
            warnings.append(f"{dataset_name}: column {col} is constant")
    return warnings
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/benchmarks/test_train_benchmark_models.py -q`
Expected: PASS with 3 passed.

- [ ] **Step 5: Expand implementation to full training stage**

```python
def main():
    ensure_benchmark_dirs()
    copy_adult_synthetic_artifacts()
    for dataset_name in ["bank", "covertype", "diabetes"]:
        spec = DATASET_REGISTRY[dataset_name]
        train_df = pd.read_csv(get_dataset_paths(dataset_name)["train"])
        ctgan = CTGAN(epochs=300, batch_size=500, verbose=True)
        ctgan.fit(train_df, discrete_columns=detect_ctgan_discrete_columns(dataset_name, train_df))
        synth_ctgan = ctgan.sample(10_000)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(train_df)
        tvae = TVAESynthesizer(metadata, epochs=300)
        tvae.fit(train_df)
        synth_tvae = tvae.sample(10_000)
        save_and_validate_synthetic(dataset_name, "ctgan", train_df, synth_ctgan, spec.target_col, spec.valid_target_values)
        save_and_validate_synthetic(dataset_name, "tvae", train_df, synth_tvae, spec.target_col, spec.valid_target_values)
        print(f"{dataset_name} - CTGAN complete ✓  |  TVAE complete ✓")
        print(f"Synthetic shape: {(10_000, len(train_df.columns))}")
        print(f"Target distribution: {pd.read_csv(get_dataset_paths(dataset_name)['ctgan'])[spec.target_col].value_counts().sort_index().to_dict()}")
    print("All models trained ✓")
```

Add dataset-by-dataset hard-failure logging, reload-after-save checks, coercion for covertype target integers if needed, and the required validation printout.

- [ ] **Step 6: Run focused tests plus the stage**

Run: `python -m pytest tests/benchmarks/test_common.py tests/benchmarks/test_train_benchmark_models.py -q`
Expected: PASS

Run: `python -m benchmarks.train_benchmark_models`
Expected: prints per-dataset completion lines, target distributions, validation status, and ends with `All models trained ✓`.

- [ ] **Step 7: Commit**

If git is available:

```bash
git add benchmarks/common.py benchmarks/train_benchmark_models.py tests/benchmarks/test_train_benchmark_models.py
git commit -m "feat: add benchmark model training"
```

If not, continue without commit.

## Task 4: Implement Cross-Domain Evaluation And Ranking

**Files:**
- Modify: `benchmarks/common.py`
- Create: `benchmarks/evaluate_benchmarks.py`
- Create: `tests/benchmarks/test_evaluate_benchmarks.py`

- [ ] **Step 1: Write the failing evaluation tests**

```python
import math

import pandas as pd

from benchmarks.evaluate_benchmarks import (
    demographic_parity_difference,
    mean_js_divergence,
    rank_models,
)


def test_demographic_parity_difference_uses_positive_rate_gap():
    df = pd.DataFrame({"age_group": ["young", "young", "older", "older"], "target": [1, 0, 1, 1]})
    assert demographic_parity_difference(df, "age_group", "target") == 0.5


def test_mean_js_divergence_is_zero_for_identical_numeric_columns():
    real_df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "target": [0, 1, 0]})
    synth_df = real_df.copy()
    assert math.isclose(mean_js_divergence(real_df, synth_df, ["x"]), 0.0, abs_tol=1e-6)


def test_rank_models_prefers_higher_tstr_and_lower_js():
    summary = pd.DataFrame(
        [
            {"Dataset": "bank", "Model": "CTGAN", "TSTR_Accuracy": 81.0, "JS_Divergence": 0.10, "MIA_Advantage": 0.11, "Demographic_Parity": 0.20},
            {"Dataset": "bank", "Model": "TVAE", "TSTR_Accuracy": 79.0, "JS_Divergence": 0.12, "MIA_Advantage": 0.15, "Demographic_Parity": 0.25},
        ]
    )
    ranked = rank_models(summary)
    assert ranked.loc[ranked["Model"] == "CTGAN", "Overall_Mean_Rank"].iloc[0] < ranked.loc[ranked["Model"] == "TVAE", "Overall_Mean_Rank"].iloc[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/benchmarks/test_evaluate_benchmarks.py -q`
Expected: FAIL because evaluation helpers are not implemented yet.

- [ ] **Step 3: Write minimal metric helpers**

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def demographic_parity_difference(df: pd.DataFrame, sensitive_col: str, target_col: str) -> float:
    groups = list(df[sensitive_col].dropna().unique())
    if len(groups) != 2:
        raise ValueError("demographic parity requires exactly two groups")
    rates = [df.loc[df[sensitive_col] == group, target_col].mean() for group in groups]
    return abs(rates[0] - rates[1])


def mean_js_divergence(real_df: pd.DataFrame, synth_df: pd.DataFrame, numeric_cols: list[str]) -> float:
    values = []
    for col in numeric_cols:
        real = real_df[col].to_numpy(dtype=float)
        synth = synth_df[col].to_numpy(dtype=float)
        bins = np.linspace(min(real.min(), synth.min()), max(real.max(), synth.max()) or 1.0, 51)
        p = np.histogram(real, bins=bins)[0].astype(float)
        q = np.histogram(synth, bins=bins)[0].astype(float)
        eps = 1e-10
        p = (p / p.sum()) + eps
        q = (q / q.sum()) + eps
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        values.append(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))
    return float(np.mean(values)) if values else float("nan")


def rank_models(summary_df: pd.DataFrame) -> pd.DataFrame:
    working = summary_df.copy()
    working["TSTR_Rank"] = working.groupby("Dataset")["TSTR_Accuracy"].rank(ascending=False, method="average")
    working["JS_Rank"] = working.groupby("Dataset")["JS_Divergence"].rank(ascending=True, method="average")
    working["MIA_Rank"] = working.groupby("Dataset")["MIA_Advantage"].rank(ascending=True, method="average")
    working["DP_Rank"] = working.groupby("Dataset")["Demographic_Parity"].rank(ascending=True, method="average")
    ranked = working.groupby("Model", as_index=False).agg(
        Mean_TSTR_Rank=("TSTR_Rank", "mean"),
        Mean_JS_Rank=("JS_Rank", "mean"),
        Mean_MIA_Rank=("MIA_Rank", "mean"),
        Mean_DP_Rank=("DP_Rank", "mean"),
    )
    ranked["Overall_Mean_Rank"] = ranked[["Mean_TSTR_Rank", "Mean_JS_Rank", "Mean_MIA_Rank", "Mean_DP_Rank"]].mean(axis=1)
    return ranked
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/benchmarks/test_evaluate_benchmarks.py -q`
Expected: PASS after correcting any implementation errors, including the final overall-rank assignment.

- [ ] **Step 5: Expand implementation to the full evaluation stage**

```python
def main():
    rows = []
    for dataset_name, spec in DATASET_REGISTRY.items():
        real_train = pd.read_csv(get_dataset_paths(dataset_name)["train"])
        real_test = pd.read_csv(get_dataset_paths(dataset_name)["test"])
        per_dataset_rows = []
        for model_name in ["CTGAN", "TVAE"]:
            synth_df = pd.read_csv(get_dataset_paths(dataset_name)[model_name.lower()])
            validate_real_and_synth_pair(dataset_name, real_train, real_test, synth_df, spec)
            result_row = {
                "Dataset": dataset_name,
                "Domain": spec.domain,
                "Fidelity_Level": spec.fidelity_level,
                "Model": model_name,
                "JS_Divergence": mean_js_divergence(real_train, synth_df, get_numeric_columns(real_train, spec.target_col)),
                "TSTR_Accuracy": tstr_accuracy(synth_df, real_test, spec.target_col),
                "MIA_Advantage": membership_inference_advantage(real_train, synth_df),
                "Demographic_Parity": (
                    demographic_parity_difference(synth_df, spec.sensitive_attr, spec.target_col)
                    if spec.sensitive_attr
                    else float("nan")
                ),
                "TSTR_Real_Baseline": tstr_accuracy(real_train, real_test, spec.target_col),
            }
            rows.append(result_row)
            per_dataset_rows.append({key: value for key, value in result_row.items() if key not in {"Dataset", "Domain", "Fidelity_Level"}})
        pd.DataFrame(per_dataset_rows).to_csv(get_dataset_paths(dataset_name)["evaluation"], index=False)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(BENCHMARK_ROOT / "results" / "cross_domain_summary.csv", index=False)
    rank_df = rank_models(summary_df)
    rank_df.to_csv(BENCHMARK_ROOT / "results" / "mean_rank_table.csv", index=False)
    print(summary_df.to_string(index=False))
```

Use shared categorical encoding, preserve `NaN` for covertype demographic parity, and verify result tables before writing and again after reloading from disk.

- [ ] **Step 6: Run focused tests plus the stage**

Run: `python -m pytest tests/benchmarks/test_common.py tests/benchmarks/test_evaluate_benchmarks.py -q`
Expected: PASS

Run: `python -m benchmarks.evaluate_benchmarks`
Expected: prints the full cross-domain summary table and writes `cross_domain_summary.csv` plus `mean_rank_table.csv`.

- [ ] **Step 7: Commit**

If git is available:

```bash
git add benchmarks/common.py benchmarks/evaluate_benchmarks.py tests/benchmarks/test_evaluate_benchmarks.py
git commit -m "feat: add cross-domain benchmark evaluation"
```

If not, continue without commit.

## Task 5: Implement Visualization And Cache-Aware Runner

**Files:**
- Modify: `benchmarks/common.py`
- Create: `benchmarks/visualize_benchmarks.py`
- Create: `benchmarks/run_benchmarks.py`
- Create: `tests/benchmarks/test_visualize_benchmarks.py`

- [ ] **Step 1: Write the failing plotting tests**

```python
from pathlib import Path

import pandas as pd

from benchmarks.visualize_benchmarks import required_plot_paths, validate_plot_inputs


def test_required_plot_paths_lists_all_four_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmarks.visualize_benchmarks.PLOTS_DIR", tmp_path)
    paths = required_plot_paths()
    assert len(paths) == 4
    assert all(path.suffix == ".png" for path in paths)


def test_validate_plot_inputs_requires_expected_columns():
    summary_df = pd.DataFrame({"Dataset": ["adult"], "Model": ["CTGAN"]})
    rank_df = pd.DataFrame({"Model": ["CTGAN"]})
    try:
        validate_plot_inputs(summary_df, rank_df)
    except ValueError as exc:
        assert "missing columns" in str(exc)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/benchmarks/test_visualize_benchmarks.py -q`
Expected: FAIL because plotting helpers do not exist yet.

- [ ] **Step 3: Write minimal plotting helper implementation**

```python
from pathlib import Path

PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def required_plot_paths() -> list[Path]:
    return [
        PLOTS_DIR / "plot1_tstr_heatmap.png",
        PLOTS_DIR / "plot2_cross_domain_dashboard.png",
        PLOTS_DIR / "plot3_mean_rank.png",
        PLOTS_DIR / "plot4_privacy_utility_all_domains.png",
    ]


def validate_plot_inputs(summary_df, rank_df) -> None:
    summary_required = {"Dataset", "Model", "TSTR_Accuracy", "MIA_Advantage", "TSTR_Real_Baseline"}
    rank_required = {"Model", "Mean_TSTR_Rank", "Mean_JS_Rank", "Mean_MIA_Rank", "Mean_DP_Rank", "Overall_Mean_Rank"}
    if not summary_required.issubset(summary_df.columns):
        raise ValueError("missing columns in summary data")
    if not rank_required.issubset(rank_df.columns):
        raise ValueError("missing columns in rank data")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/benchmarks/test_visualize_benchmarks.py -q`
Expected: PASS with 2 passed.

- [ ] **Step 5: Expand implementation to the full plotting and runner behavior**

```python
def main():
    summary_df = pd.read_csv(RESULTS_DIR / "cross_domain_summary.csv")
    rank_df = pd.read_csv(RESULTS_DIR / "mean_rank_table.csv")
    validate_plot_inputs(summary_df, rank_df)
    plt.style.use("seaborn-v0_8-whitegrid")
    build_tstr_heatmap(summary_df, PLOTS_DIR / "plot1_tstr_heatmap.png")
    build_metric_dashboard(summary_df, PLOTS_DIR / "plot2_cross_domain_dashboard.png")
    build_mean_rank_chart(rank_df, PLOTS_DIR / "plot3_mean_rank.png")
    build_privacy_utility_scatter(summary_df, PLOTS_DIR / "plot4_privacy_utility_all_domains.png")
    print("Plot 1 saved ✓")
    print("Plot 2 saved ✓")
    print("Plot 3 saved ✓")
    print("Plot 4 saved ✓")
```

```python
def main():
    print("=" * 65)
    print("   CROSS-DOMAIN BENCHMARKING PIPELINE")
    print("   4 Datasets x 2 Models x 4 Metrics")
    print("=" * 65)
    run_stage_if_missing("download", expected_dataset_files(), "benchmarks.download_datasets")
    run_stage_if_missing("training", expected_synthetic_files(non_adult_only=True), "benchmarks.train_benchmark_models")
    run_stage_if_missing("evaluation", [RESULTS_DIR / "cross_domain_summary.csv"], "benchmarks.evaluate_benchmarks")
    run_stage_if_missing("plots", required_plot_paths(), "benchmarks.visualize_benchmarks")
    print(pd.read_csv(RESULTS_DIR / "cross_domain_summary.csv").to_string(index=False))
    print(pd.read_csv(RESULTS_DIR / "mean_rank_table.csv").to_string(index=False))
```

The runner must check complete artifact sets rather than single files before skipping a stage, and it must never be used until the four manual scripts have already succeeded.

- [ ] **Step 6: Run focused tests plus the stage**

Run: `python -m pytest tests/benchmarks/test_common.py tests/benchmarks/test_visualize_benchmarks.py -q`
Expected: PASS

Run: `python -m benchmarks.visualize_benchmarks`
Expected: writes four PNGs and prints four saved messages.

- [ ] **Step 7: Commit**

If git is available:

```bash
git add benchmarks/common.py benchmarks/visualize_benchmarks.py benchmarks/run_benchmarks.py tests/benchmarks/test_visualize_benchmarks.py
git commit -m "feat: add benchmark visualizations and runner"
```

If not, continue without commit.

## Task 6: End-To-End Verification And Required Manual Execution Order

**Files:**
- Modify: none unless verification reveals defects

- [ ] **Step 1: Run the benchmark unit tests**

Run: `python -m pytest tests/benchmarks -q`
Expected: PASS with all benchmark tests green.

- [ ] **Step 2: Run the required command 1**

Run: `python -m benchmarks.download_datasets`
Expected: final output includes the summary table. Capture and share the output.

- [ ] **Step 3: Run the required command 2**

Run: `python -m benchmarks.train_benchmark_models`
Expected: final output includes `All models trained ✓`. Capture and share the output.

- [ ] **Step 4: Run the required command 3**

Run: `python -m benchmarks.evaluate_benchmarks`
Expected: prints the full cross-domain summary table. Capture and share the output.

- [ ] **Step 5: Run the required command 4**

Run: `python -m benchmarks.visualize_benchmarks`
Expected: prints four plot-saved lines and creates four PNG files. Share the images.

- [ ] **Step 6: Optional cache check after success**

Run: `python -m benchmarks.run_benchmarks`
Expected: only after the first four commands succeed, confirms caching behavior and prints final summary tables.

- [ ] **Step 7: Final verification sweep**

Run:

```bash
python -m pytest tests/benchmarks -q
python -m benchmarks.evaluate_benchmarks
python -m benchmarks.visualize_benchmarks
```

Expected: all commands exit 0 and outputs match saved artifacts. If any differ, fix the code before making completion claims.
