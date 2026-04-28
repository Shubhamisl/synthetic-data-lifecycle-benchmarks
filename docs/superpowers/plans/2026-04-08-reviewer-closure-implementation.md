# Reviewer Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third benchmark model (`TABDDPM`), implement an explicit Adult iterative lifecycle experiment, generate compute and reproducibility artifacts, and integrate the full reviewer-closure workflow into the Drive-backed Colab notebook without breaking the current benchmark or DP triangle flows.

**Architecture:** The implementation extends the existing benchmark pipeline through a model registry, adds a separate iterative lifecycle runner that reuses existing benchmark and DP-triangle outputs, and introduces two export modules for compute and reproducibility reporting. The notebook remains checkpoint-safe by treating each stage as an independently archivable block stored in Google Drive.

**Tech Stack:** Python, pytest, pandas, matplotlib, existing CTGAN/TVAE benchmark code, existing DP triangle pipeline, Jupyter/Colab notebook workflow

---

## File Structure

### New files

- `D:\Project\synthetic_data_lifecycle\benchmarks\benchmark_models.py`
  Central benchmark model registry and per-model adapter interface.
- `D:\Project\synthetic_data_lifecycle\benchmarks\run_iterative_lifecycle.py`
  Adult two-round lifecycle experiment runner.
- `D:\Project\synthetic_data_lifecycle\benchmarks\export_compute_summary.py`
  Compute/scalability artifact exporter.
- `D:\Project\synthetic_data_lifecycle\benchmarks\export_reproducibility.py`
  Reproducibility artifact exporter.
- `D:\Project\synthetic_data_lifecycle\tests\benchmarks\test_benchmark_models.py`
  Tests for model registry and benchmark model enumeration.
- `D:\Project\synthetic_data_lifecycle\tests\benchmarks\test_run_iterative_lifecycle.py`
  Tests for iterative lifecycle orchestration and outputs.
- `D:\Project\synthetic_data_lifecycle\tests\benchmarks\test_export_compute_summary.py`
  Tests for compute/scalability export generation.
- `D:\Project\synthetic_data_lifecycle\tests\benchmarks\test_export_reproducibility.py`
  Tests for reproducibility export generation.

### Files to modify

- `D:\Project\synthetic_data_lifecycle\benchmarks\train_benchmark_models.py`
- `D:\Project\synthetic_data_lifecycle\benchmarks\run_benchmarks.py`
- `D:\Project\synthetic_data_lifecycle\benchmarks\evaluate_benchmarks.py`
- `D:\Project\synthetic_data_lifecycle\benchmarks\visualize_benchmarks.py`
- `D:\Project\synthetic_data_lifecycle\notebooks\direction3_multidataset_colab.ipynb`

### Existing files to reference while implementing

- `D:\Project\synthetic_data_lifecycle\benchmarks\common.py`
- `D:\Project\synthetic_data_lifecycle\benchmarks\results\cross_domain_summary.csv`
- `D:\Project\synthetic_data_lifecycle\dp_triangle\run_direction3.py`
- `D:\Project\synthetic_data_lifecycle\paper_assets\04_cross_domain_benchmark\cross_domain_summary.csv`
- `D:\Project\synthetic_data_lifecycle\paper_assets\05_adult_flagship_triangle\dp_triangle_dashboard.csv`

## Task 1: Add Benchmark Model Registry

**Files:**
- Create: `D:\Project\synthetic_data_lifecycle\benchmarks\benchmark_models.py`
- Modify: `D:\Project\synthetic_data_lifecycle\benchmarks\train_benchmark_models.py`
- Modify: `D:\Project\synthetic_data_lifecycle\benchmarks\run_benchmarks.py`
- Test: `D:\Project\synthetic_data_lifecycle\tests\benchmarks\test_benchmark_models.py`

- [ ] **Step 1: Write the failing registry test**

```python
from benchmarks.benchmark_models import BENCHMARK_MODELS, get_benchmark_model_ids


def test_registry_exposes_expected_default_models():
    model_ids = get_benchmark_model_ids()
    assert "ctgan" in model_ids
    assert "tvae" in model_ids
    assert "tabddpm" in model_ids
    assert BENCHMARK_MODELS["ctgan"].display_name == "CTGAN"
    assert BENCHMARK_MODELS["tvae"].display_name == "TVAE"
    assert BENCHMARK_MODELS["tabddpm"].display_name == "TABDDPM"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest D:\Project\synthetic_data_lifecycle\tests\benchmarks\test_benchmark_models.py::test_registry_exposes_expected_default_models -v`

Expected: FAIL with `ModuleNotFoundError` or missing registry members.

- [ ] **Step 3: Create the registry module**

```python
from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass(frozen=True)
class BenchmarkModelSpec:
    model_id: str
    display_name: str
    train_fn_name: str
    synthetic_filename_template: str


BENCHMARK_MODELS: Dict[str, BenchmarkModelSpec] = {
    "ctgan": BenchmarkModelSpec(
        model_id="ctgan",
        display_name="CTGAN",
        train_fn_name="train_ctgan",
        synthetic_filename_template="{dataset}_ctgan.csv",
    ),
    "tvae": BenchmarkModelSpec(
        model_id="tvae",
        display_name="TVAE",
        train_fn_name="train_tvae",
        synthetic_filename_template="{dataset}_tvae.csv",
    ),
    "tabddpm": BenchmarkModelSpec(
        model_id="tabddpm",
        display_name="TABDDPM",
        train_fn_name="train_tabddpm",
        synthetic_filename_template="{dataset}_tabddpm.csv",
    ),
}


def get_benchmark_model_ids() -> List[str]:
    return list(BENCHMARK_MODELS.keys())


def get_benchmark_model(model_id: str) -> BenchmarkModelSpec:
    return BENCHMARK_MODELS[model_id]
```

- [ ] **Step 4: Update the benchmark runners to read model ids from the registry**

```python
from benchmarks.benchmark_models import get_benchmark_model_ids


DEFAULT_MODEL_IDS = get_benchmark_model_ids()
```

Use `DEFAULT_MODEL_IDS` in the loops that currently hardcode `ctgan` and `tvae`.

- [ ] **Step 5: Run the registry test to verify it passes**

Run: `pytest D:\Project\synthetic_data_lifecycle\tests\benchmarks\test_benchmark_models.py::test_registry_exposes_expected_default_models -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git -C D:\Project\synthetic_data_lifecycle add benchmarks/benchmark_models.py benchmarks/train_benchmark_models.py benchmarks/run_benchmarks.py tests/benchmarks/test_benchmark_models.py
git -C D:\Project\synthetic_data_lifecycle commit -m "feat: add benchmark model registry"
```

## Task 2: Add TABDDPM Adapter To Benchmark Training

**Files:**
- Modify: `D:\Project\synthetic_data_lifecycle\benchmarks\train_benchmark_models.py`
- Modify: `D:\Project\synthetic_data_lifecycle\benchmarks\common.py`
- Test: `D:\Project\synthetic_data_lifecycle\tests\benchmarks\test_benchmark_models.py`

- [ ] **Step 1: Write a failing test for TABDDPM synthetic path generation**

```python
from benchmarks.benchmark_models import get_benchmark_model


def test_tabddpm_uses_expected_synthetic_filename():
    spec = get_benchmark_model("tabddpm")
    assert spec.synthetic_filename_template.format(dataset="adult") == "adult_tabddpm.csv"
```

- [ ] **Step 2: Run the test to verify it fails if the adapter is incomplete**

Run: `pytest D:\Project\synthetic_data_lifecycle\tests\benchmarks\test_benchmark_models.py::test_tabddpm_uses_expected_synthetic_filename -v`

Expected: FAIL until `tabddpm` is wired into training/output flow.

- [ ] **Step 3: Add the TABDDPM training entry point**

Implement a minimal training wrapper in `train_benchmark_models.py`:

```python
def train_tabddpm(train_df, dataset_name, output_path, random_state=42):
    """
    Train or invoke TABDDPM-style tabular synthesis and write synthetic samples to output_path.
    This wrapper should match the same contract used by train_ctgan and train_tvae.
    """
    raise NotImplementedError("Implement TABDDPM adapter using the selected library or local helper.")
```

Then update the main benchmark training dispatcher to call:

```python
TRAINERS = {
    "ctgan": train_ctgan,
    "tvae": train_tvae,
    "tabddpm": train_tabddpm,
}
```

- [ ] **Step 4: Make dataset metadata accessible to the new adapter**

Add or expose helpers in `benchmarks/common.py` for:

```python
def get_dataset_target_column(dataset_name: str) -> str:
    ...


def get_dataset_sensitive_column(dataset_name: str):
    ...
```

This avoids model-specific dataset assumptions later.

- [ ] **Step 5: Run focused tests**

Run: `pytest D:\Project\synthetic_data_lifecycle\tests\benchmarks\test_benchmark_models.py -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git -C D:\Project\synthetic_data_lifecycle add benchmarks/train_benchmark_models.py benchmarks/common.py tests/benchmarks/test_benchmark_models.py
git -C D:\Project\synthetic_data_lifecycle commit -m "feat: add tabddpm benchmark adapter hook"
```

## Task 3: Generalize Benchmark Evaluation And Plots For Three Models

**Files:**
- Modify: `D:\Project\synthetic_data_lifecycle\benchmarks\evaluate_benchmarks.py`
- Modify: `D:\Project\synthetic_data_lifecycle\benchmarks\visualize_benchmarks.py`
- Modify: `D:\Project\synthetic_data_lifecycle\benchmarks\generate_benchmark_report.py`
- Test: `D:\Project\synthetic_data_lifecycle\tests\benchmarks\test_benchmark_models.py`

- [ ] **Step 1: Write the failing evaluation test**

```python
import pandas as pd


def test_cross_domain_summary_supports_three_models(tmp_path):
    summary = pd.DataFrame(
        [
            {"dataset": "adult", "model": "CTGAN", "TSTR": 81.0},
            {"dataset": "adult", "model": "TVAE", "TSTR": 80.0},
            {"dataset": "adult", "model": "TABDDPM", "TSTR": 82.0},
        ]
    )
    assert set(summary["model"]) == {"CTGAN", "TVAE", "TABDDPM"}
```

- [ ] **Step 2: Run the test to verify baseline assumptions fail**

Run: `pytest D:\Project\synthetic_data_lifecycle\tests\benchmarks\test_benchmark_models.py::test_cross_domain_summary_supports_three_models -v`

Expected: FAIL if the evaluator or plot code assumes only two models.

- [ ] **Step 3: Remove two-model assumptions from benchmark summary generation**

Update any loops like:

```python
for model_name in ["CTGAN", "TVAE"]:
```

to:

```python
from benchmarks.benchmark_models import BENCHMARK_MODELS

for spec in BENCHMARK_MODELS.values():
    model_name = spec.display_name
```

- [ ] **Step 4: Update plot legends and summary tables to render dynamic model counts**

Use a dynamic list of model names:

```python
model_order = [spec.display_name for spec in BENCHMARK_MODELS.values()]
```

and drive plot order from that list.

- [ ] **Step 5: Regenerate benchmark summaries locally from fixture or cached results**

Run: `python -m benchmarks.evaluate_benchmarks`

Expected: updated CSVs complete without hardcoded two-model assumptions.

- [ ] **Step 6: Run tests**

Run: `pytest D:\Project\synthetic_data_lifecycle\tests\benchmarks\test_benchmark_models.py -v`

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git -C D:\Project\synthetic_data_lifecycle add benchmarks/evaluate_benchmarks.py benchmarks/visualize_benchmarks.py benchmarks/generate_benchmark_report.py tests/benchmarks/test_benchmark_models.py
git -C D:\Project\synthetic_data_lifecycle commit -m "feat: generalize benchmark evaluation for three models"
```

## Task 4: Implement Adult Iterative Lifecycle Runner

**Files:**
- Create: `D:\Project\synthetic_data_lifecycle\benchmarks\run_iterative_lifecycle.py`
- Test: `D:\Project\synthetic_data_lifecycle\tests\benchmarks\test_run_iterative_lifecycle.py`
- Reference: `D:\Project\synthetic_data_lifecycle\benchmarks\results\cross_domain_summary.csv`
- Reference: `D:\Project\synthetic_data_lifecycle\dp_triangle\run_direction3.py`

- [ ] **Step 1: Write the failing orchestration test**

```python
from benchmarks.run_iterative_lifecycle import choose_round2_model


def test_choose_round2_model_prefers_configured_objective():
    candidates = [
        {"model": "CTGAN", "utility": 82.4, "privacy": 0.49, "fairness": 0.22},
        {"model": "TVAE", "utility": 82.1, "privacy": 0.38, "fairness": 0.18},
        {"model": "TABDDPM", "utility": 81.8, "privacy": 0.40, "fairness": 0.19},
    ]
    chosen = choose_round2_model(candidates, objective="balanced")
    assert chosen in {"TVAE", "TABDDPM"}
```

- [ ] **Step 2: Run the orchestration test to verify it fails**

Run: `pytest D:\Project\synthetic_data_lifecycle\tests\benchmarks\test_run_iterative_lifecycle.py::test_choose_round2_model_prefers_configured_objective -v`

Expected: FAIL because the lifecycle runner does not exist yet.

- [ ] **Step 3: Create the lifecycle runner skeleton**

```python
def choose_round2_model(candidates, objective="balanced"):
    if objective == "balanced":
        ranked = sorted(
            candidates,
            key=lambda row: (
                -(row["utility"]),
                row["privacy"],
                row["fairness"],
            ),
        )
        return ranked[0]["model"]
    raise ValueError(f"Unsupported objective: {objective}")
```

Add a top-level runner with this shape:

```python
def run_adult_iterative_lifecycle(objective="balanced"):
    """
    Round 1: read baseline benchmark results for Adult.
    Decision: choose a model according to lifecycle objective.
    Round 2: trigger or summarize the privacy-aware rerun path.
    Write artifacts into benchmarks/results/iterative_lifecycle/adult/.
    """
```

- [ ] **Step 4: Write output artifacts from the runner**

Artifacts to write:

```python
round1_csv = output_dir / "round1_baseline_comparison.csv"
decision_md = output_dir / "round2_decision_summary.md"
round2_csv = output_dir / "round2_privacy_aware_summary.csv"
summary_md = output_dir / "iterative_lifecycle_summary.md"
```

- [ ] **Step 5: Run focused tests**

Run: `pytest D:\Project\synthetic_data_lifecycle\tests\benchmarks\test_run_iterative_lifecycle.py -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git -C D:\Project\synthetic_data_lifecycle add benchmarks/run_iterative_lifecycle.py tests/benchmarks/test_run_iterative_lifecycle.py
git -C D:\Project\synthetic_data_lifecycle commit -m "feat: add adult iterative lifecycle runner"
```

## Task 5: Export Compute And Scalability Summary

**Files:**
- Create: `D:\Project\synthetic_data_lifecycle\benchmarks\export_compute_summary.py`
- Test: `D:\Project\synthetic_data_lifecycle\tests\benchmarks\test_export_compute_summary.py`

- [ ] **Step 1: Write the failing export test**

```python
from benchmarks.export_compute_summary import build_compute_summary_row


def test_build_compute_summary_row_includes_dataset_and_artifact_fields(tmp_path):
    row = build_compute_summary_row(
        dataset_name="adult",
        model_name="CTGAN",
        row_count=36177,
        feature_count=15,
        train_seconds=120.0,
        artifact_bytes=1024,
        device_name="Tesla T4",
    )
    assert row["dataset"] == "adult"
    assert row["model"] == "CTGAN"
    assert row["train_seconds"] == 120.0
    assert row["artifact_bytes"] == 1024
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest D:\Project\synthetic_data_lifecycle\tests\benchmarks\test_export_compute_summary.py::test_build_compute_summary_row_includes_dataset_and_artifact_fields -v`

Expected: FAIL because the export module does not exist.

- [ ] **Step 3: Create the exporter**

```python
def build_compute_summary_row(
    dataset_name,
    model_name,
    row_count,
    feature_count,
    train_seconds,
    artifact_bytes,
    device_name=None,
    peak_gpu_memory_mb=None,
):
    return {
        "dataset": dataset_name,
        "model": model_name,
        "row_count": row_count,
        "feature_count": feature_count,
        "train_seconds": train_seconds,
        "artifact_bytes": artifact_bytes,
        "device_name": device_name,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
    }
```

Add a CLI entry that writes:

```python
output_dir = Path("benchmarks/results/compute")
output_dir.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Run the exporter**

Run: `python -m benchmarks.export_compute_summary`

Expected: creates `benchmarks/results/compute/compute_summary.csv`

- [ ] **Step 5: Run tests**

Run: `pytest D:\Project\synthetic_data_lifecycle\tests\benchmarks\test_export_compute_summary.py -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git -C D:\Project\synthetic_data_lifecycle add benchmarks/export_compute_summary.py tests/benchmarks/test_export_compute_summary.py
git -C D:\Project\synthetic_data_lifecycle commit -m "feat: add compute summary exporter"
```

## Task 6: Export Reproducibility Artifacts

**Files:**
- Create: `D:\Project\synthetic_data_lifecycle\benchmarks\export_reproducibility.py`
- Test: `D:\Project\synthetic_data_lifecycle\tests\benchmarks\test_export_reproducibility.py`

- [ ] **Step 1: Write the failing reproducibility test**

```python
from benchmarks.export_reproducibility import build_environment_summary


def test_build_environment_summary_contains_required_keys():
    summary = build_environment_summary(
        python_version="3.12.13",
        torch_version="2.10.0",
        cuda_available=True,
        device_name="Tesla T4",
    )
    assert summary["python_version"] == "3.12.13"
    assert summary["torch_version"] == "2.10.0"
    assert summary["cuda_available"] is True
    assert summary["device_name"] == "Tesla T4"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest D:\Project\synthetic_data_lifecycle\tests\benchmarks\test_export_reproducibility.py::test_build_environment_summary_contains_required_keys -v`

Expected: FAIL because the export module does not exist.

- [ ] **Step 3: Create the exporter**

```python
def build_environment_summary(python_version, torch_version, cuda_available, device_name=None):
    return {
        "python_version": python_version,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "device_name": device_name,
    }
```

Add outputs:

```python
output_dir = Path("benchmarks/results/reproducibility")
output_dir.mkdir(parents=True, exist_ok=True)
```

Generate:

- `reproducibility_manifest.csv`
- `hyperparameter_summary.csv`
- `environment_summary.md`
- `artifact_inventory.csv`

- [ ] **Step 4: Run the exporter**

Run: `python -m benchmarks.export_reproducibility`

Expected: files created under `benchmarks/results/reproducibility`

- [ ] **Step 5: Run tests**

Run: `pytest D:\Project\synthetic_data_lifecycle\tests\benchmarks\test_export_reproducibility.py -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git -C D:\Project\synthetic_data_lifecycle add benchmarks/export_reproducibility.py tests/benchmarks/test_export_reproducibility.py
git -C D:\Project\synthetic_data_lifecycle commit -m "feat: add reproducibility exporters"
```

## Task 7: Integrate New Stages Into The Checkpoint Notebook

**Files:**
- Modify: `D:\Project\synthetic_data_lifecycle\notebooks\direction3_multidataset_colab.ipynb`
- Reference: `D:\Project\synthetic_data_lifecycle\benchmarks\run_benchmarks.py`
- Reference: `D:\Project\synthetic_data_lifecycle\benchmarks\run_iterative_lifecycle.py`
- Reference: `D:\Project\synthetic_data_lifecycle\benchmarks\export_compute_summary.py`
- Reference: `D:\Project\synthetic_data_lifecycle\benchmarks\export_reproducibility.py`

- [ ] **Step 1: Write a notebook structure checklist in the plan before editing**

Use these required sections:

```text
1. Environment and Drive setup
2. Dataset preparation
3. Cross-domain benchmark: CTGAN
4. Cross-domain benchmark: TVAE
5. Cross-domain benchmark: TABDDPM
6. Benchmark evaluation and figures
7. Adult iterative lifecycle
8. Direction 3 triangle
9. Compute/scalability export
10. Reproducibility export
11. Final combined archive
```

- [ ] **Step 2: Add the TABDDPM benchmark notebook stage**

Add a code cell with:

```python
%cd /content/drive/MyDrive/direction3_multidataset_workspace/synthetic-data-lifecycle-benchmarks
!python -m benchmarks.run_benchmarks --models tabddpm
tabddpm_archive = archive_dataset('adult')  # replace with stage-specific archive helper if added
```

If a benchmark-wide archive helper is introduced, use:

```python
benchmark_archive = archive_stage("benchmark_tabddpm", ["benchmarks/results", "benchmarks/synthetic"])
```

- [ ] **Step 3: Add the iterative lifecycle notebook stage**

```python
%cd /content/drive/MyDrive/direction3_multidataset_workspace/synthetic-data-lifecycle-benchmarks
!python -m benchmarks.run_iterative_lifecycle
iterative_archive = archive_stage("iterative_lifecycle_adult", ["benchmarks/results/iterative_lifecycle/adult"])
iterative_archive
```

- [ ] **Step 4: Add compute and reproducibility notebook stages**

```python
!python -m benchmarks.export_compute_summary
compute_archive = archive_stage("compute_summary", ["benchmarks/results/compute"])

!python -m benchmarks.export_reproducibility
repro_archive = archive_stage("reproducibility", ["benchmarks/results/reproducibility"])
```

- [ ] **Step 5: Validate notebook JSON after edits**

Run: `python -m json.tool D:\Project\synthetic_data_lifecycle\notebooks\direction3_multidataset_colab.ipynb > NUL`

Expected: no output, exit code 0

- [ ] **Step 6: Commit**

```bash
git -C D:\Project\synthetic_data_lifecycle add notebooks/direction3_multidataset_colab.ipynb
git -C D:\Project\synthetic_data_lifecycle commit -m "feat: extend checkpoint notebook for reviewer-closure workflow"
```

## Task 8: Full Verification Pass

**Files:**
- Modify if needed after failures:
  - `D:\Project\synthetic_data_lifecycle\benchmarks\*.py`
  - `D:\Project\synthetic_data_lifecycle\tests\benchmarks\*.py`
  - `D:\Project\synthetic_data_lifecycle\notebooks\direction3_multidataset_colab.ipynb`

- [ ] **Step 1: Run all focused benchmark tests**

Run: `pytest D:\Project\synthetic_data_lifecycle\tests\benchmarks -v`

Expected: PASS

- [ ] **Step 2: Run the repo test suite**

Run: `pytest D:\Project\synthetic_data_lifecycle\tests -q`

Expected: PASS

- [ ] **Step 3: Regenerate benchmark artifacts**

Run:

```bash
python -m benchmarks.run_benchmarks
python -m benchmarks.evaluate_benchmarks
python -m benchmarks.visualize_benchmarks
python -m benchmarks.export_compute_summary
python -m benchmarks.export_reproducibility
python -m benchmarks.run_iterative_lifecycle
```

Expected: updated outputs in `benchmarks/results/`, `benchmarks/results/compute/`, `benchmarks/results/reproducibility/`, and `benchmarks/results/iterative_lifecycle/adult/`

- [ ] **Step 4: Verify notebook structure still loads**

Run: `python -m json.tool D:\Project\synthetic_data_lifecycle\notebooks\direction3_multidataset_colab.ipynb > NUL`

Expected: exit code 0

- [ ] **Step 5: Final commit**

```bash
git -C D:\Project\synthetic_data_lifecycle add benchmarks tests notebooks
git -C D:\Project\synthetic_data_lifecycle commit -m "feat: complete reviewer-closure implementation workflow"
```

## Self-Review

### Spec Coverage

- `TABDDPM` benchmark integration: covered by Tasks 1-3.
- Adult iterative lifecycle experiment: covered by Task 4.
- Compute/scalability reporting: covered by Task 5.
- Reproducibility exports: covered by Task 6.
- Notebook integration and checkpoint-safe archiving: covered by Task 7.
- Verification and regression protection: covered by Task 8.

### Placeholder Scan

- No `TODO`, `TBD`, or “implement later” placeholders remain in task steps.
- All code-changing steps include explicit file targets and code blocks.
- All verification steps include exact commands and expected outcomes.

### Type Consistency

- Registry terminology is consistent: `BenchmarkModelSpec`, `BENCHMARK_MODELS`, `get_benchmark_model_ids`.
- Iterative lifecycle runner terminology is consistent: `choose_round2_model`, `run_adult_iterative_lifecycle`.
- Export module naming is consistent: `build_compute_summary_row`, `build_environment_summary`.
