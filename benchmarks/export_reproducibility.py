from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .benchmark_models import get_benchmark_models
from .common import BENCHMARK_ROOT, DATASET_REGISTRY

RESULTS_DIR = BENCHMARK_ROOT / "results" / "reproducibility"


@dataclass(frozen=True)
class FileRecord:
    category: str
    name: str
    path: Path
    present: bool
    bytes: int | None
    mtime_utc: str


def _utc_timestamp(path: Path | None) -> str:
    if path is None or not path.exists():
        return "N/A"
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _file_stats(path: Path | None) -> tuple[bool, int | None, str]:
    if path is None or not path.exists():
        return False, None, "N/A"
    return True, path.stat().st_size, _utc_timestamp(path)


def _read_csv_shape(path: Path | None) -> tuple[int | None, int | None]:
    if path is None or not path.exists():
        return None, None
    try:
        dataframe = pd.read_csv(path)
    except Exception:
        return None, None
    return int(dataframe.shape[0]), int(dataframe.shape[1])


def _render_markdown_table(dataframe: pd.DataFrame, columns: list[str]) -> str:
    table = dataframe.loc[:, columns].copy()

    def stringify(value: object) -> str:
        if value is None:
            return "N/A"
        if pd.isna(value):
            return "N/A"
        if isinstance(value, bool):
            return "Yes" if value else "No"
        return str(value)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for row in table.itertuples(index=False):
        rows.append("| " + " | ".join(stringify(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join([header, separator, *rows])


def _git_metadata(project_root: Path) -> dict[str, str]:
    metadata = {"git_branch": "N/A", "git_commit": "N/A"}
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return metadata

    metadata["git_branch"] = branch or "N/A"
    metadata["git_commit"] = commit or "N/A"
    return metadata


def _package_version(package_name: str) -> str:
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        return "N/A"


def _discover_outputs(benchmark_root: Path, project_root: Path) -> list[FileRecord]:
    files: list[tuple[str, str, Path]] = []

    for dataset_name in DATASET_REGISTRY:
        files.extend(
            [
                ("dataset", f"{dataset_name} train", benchmark_root / "datasets" / f"{dataset_name}_train.csv"),
                ("dataset", f"{dataset_name} test", benchmark_root / "datasets" / f"{dataset_name}_test.csv"),
            ]
        )
        for model_spec in get_benchmark_models():
            files.append(
                (
                    "synthetic",
                    f"{dataset_name} {model_spec.display_name}",
                    benchmark_root / "synthetic" / f"{dataset_name}_{model_spec.model_id}.csv",
                )
            )

    files.extend(
        [
            ("benchmark_results", "cross-domain summary", benchmark_root / "results" / "cross_domain_summary.csv"),
            ("benchmark_results", "mean rank table", benchmark_root / "results" / "mean_rank_table.csv"),
            ("benchmark_results", "benchmark run notes", benchmark_root / "results" / "benchmark_run_notes.md"),
            ("benchmark_results", "benchmark research summary", benchmark_root / "results" / "benchmark_research_summary.md"),
            ("benchmark_results", "benchmark failures log", benchmark_root / "results" / "benchmark_failures.log"),
            ("benchmark_results", "cross-domain report", benchmark_root / "results" / "cross_domain_benchmark_report.docx"),
            ("benchmark_results", "compute summary", benchmark_root / "results" / "compute" / "compute_summary.csv"),
            ("benchmark_results", "compute markdown", benchmark_root / "results" / "compute" / "compute_summary.md"),
            ("benchmark_results", "reproducibility manifest", benchmark_root / "results" / "reproducibility" / "reproducibility_manifest.csv"),
            ("benchmark_results", "artifact inventory", benchmark_root / "results" / "reproducibility" / "artifact_inventory.csv"),
            ("dp_triangle", "adult dp triangle dashboard", project_root / "colab_artifacts" / "results" / "dp_triangle_dashboard.csv"),
            ("dp_triangle", "adult direction3 findings", project_root / "colab_artifacts" / "results" / "direction3_final_findings.md"),
            ("dp_triangle", "adult direction3 methodology", project_root / "colab_artifacts" / "results" / "direction3_methodology_note.md"),
        ]
    )

    records: list[FileRecord] = []
    for category, name, path in files:
        present, bytes_, mtime_utc = _file_stats(path)
        records.append(FileRecord(category, name, path, present, bytes_, mtime_utc))
    return records


def build_dataset_manifest(benchmark_root: Path | None = None) -> pd.DataFrame:
    root = benchmark_root or BENCHMARK_ROOT
    rows: list[dict[str, object]] = []
    for dataset_name, spec in DATASET_REGISTRY.items():
        train_path = root / "datasets" / f"{dataset_name}_train.csv"
        test_path = root / "datasets" / f"{dataset_name}_test.csv"
        train_rows, train_features = _read_csv_shape(train_path)
        test_rows, test_features = _read_csv_shape(test_path)
        train_present, train_bytes, train_mtime = _file_stats(train_path)
        test_present, test_bytes, test_mtime = _file_stats(test_path)

        rows.append(
            {
                "dataset_name": dataset_name,
                "domain": spec.domain,
                "target_column": spec.target_col,
                "sensitive_attr": spec.sensitive_attr or "N/A",
                "train_rows": train_rows,
                "train_features": train_features,
                "train_present": train_present,
                "train_bytes": train_bytes,
                "train_mtime_utc": train_mtime,
                "test_rows": test_rows,
                "test_features": test_features,
                "test_present": test_present,
                "test_bytes": test_bytes,
                "test_mtime_utc": test_mtime,
            }
        )

    return pd.DataFrame(rows)


def build_model_manifest(benchmark_root: Path | None = None) -> pd.DataFrame:
    root = benchmark_root or BENCHMARK_ROOT
    project_root = root.parent
    rows: list[dict[str, object]] = []
    for model_spec in get_benchmark_models():
        adult_source = model_spec.adult_source_path
        if adult_source is not None and not adult_source.is_absolute():
            adult_source = project_root / adult_source
        adult_present, adult_bytes, adult_mtime = _file_stats(adult_source)

        rows.append(
            {
                "model_id": model_spec.model_id,
                "display_name": model_spec.display_name,
                "available": model_spec.available,
                "trainable": model_spec.trainable,
                "adult_source_path": str(adult_source) if adult_source is not None else "N/A",
                "adult_source_present": adult_present,
                "adult_source_bytes": adult_bytes,
                "adult_source_mtime_utc": adult_mtime,
            }
        )

    return pd.DataFrame(rows)


def build_runtime_manifest(benchmark_root: Path | None = None) -> pd.DataFrame:
    root = benchmark_root or BENCHMARK_ROOT
    project_root = root.parent
    git_metadata = _git_metadata(project_root)
    rows = [
        {"key": "generated_utc", "value": datetime.now(timezone.utc).isoformat()},
        {"key": "python_version", "value": platform.python_version()},
        {"key": "python_executable", "value": sys.executable},
        {"key": "platform", "value": platform.platform()},
        {"key": "system", "value": platform.system()},
        {"key": "release", "value": platform.release()},
        {"key": "machine", "value": platform.machine()},
        {"key": "processor", "value": platform.processor() or "N/A"},
        {"key": "cwd", "value": str(Path.cwd())},
        {"key": "timezone", "value": os.environ.get("TZ", "N/A")},
        {"key": "cpu_count", "value": os.cpu_count() or "N/A"},
        {"key": "pandas_version", "value": _package_version("pandas")},
        {"key": "numpy_version", "value": _package_version("numpy")},
        {"key": "matplotlib_version", "value": _package_version("matplotlib")},
        {"key": "git_branch", "value": git_metadata["git_branch"]},
        {"key": "git_commit", "value": git_metadata["git_commit"]},
    ]

    try:
        import torch

        rows.append({"key": "torch_version", "value": getattr(torch, "__version__", "N/A")})
        rows.append({"key": "cuda_available", "value": bool(torch.cuda.is_available())})
        rows.append({"key": "cuda_device_count", "value": torch.cuda.device_count()})
    except Exception:
        rows.append({"key": "torch_version", "value": "N/A"})
        rows.append({"key": "cuda_available", "value": "N/A"})
        rows.append({"key": "cuda_device_count", "value": "N/A"})

    return pd.DataFrame(rows)


def build_artifact_inventory(benchmark_root: Path | None = None) -> pd.DataFrame:
    root = benchmark_root or BENCHMARK_ROOT
    project_root = root.parent
    records = _discover_outputs(root, project_root)
    return pd.DataFrame(
        [
            {
                "category": record.category,
                "name": record.name,
                "path": str(record.path),
                "present": record.present,
                "bytes": record.bytes,
                "mtime_utc": record.mtime_utc,
            }
            for record in records
        ]
    )


def build_reproducibility_manifest(benchmark_root: Path | None = None) -> pd.DataFrame:
    root = benchmark_root or BENCHMARK_ROOT
    project_root = root.parent
    dataset_df = build_dataset_manifest(root)
    model_df = build_model_manifest(root)
    runtime_df = build_runtime_manifest(root)
    inventory_df = build_artifact_inventory(root)

    rows: list[dict[str, object]] = []
    for row in dataset_df.to_dict(orient="records"):
        rows.append({"section": "dataset", **row})
    for row in model_df.to_dict(orient="records"):
        rows.append({"section": "model", **row})
    for row in runtime_df.to_dict(orient="records"):
        rows.append({"section": "runtime", **row})
    for row in inventory_df.to_dict(orient="records"):
        rows.append({"section": "artifact", **row})

    rows.append(
        {
            "section": "context",
            "dataset_count": len(dataset_df),
            "model_count": len(model_df),
            "artifact_count": len(inventory_df),
            "project_root": str(project_root),
            "benchmark_root": str(root),
        }
    )

    return pd.DataFrame(rows)


def write_reproducibility_exports(benchmark_root: Path | None = None) -> dict[str, Path]:
    root = benchmark_root or BENCHMARK_ROOT
    output_dir = root / "results" / "reproducibility"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_df = build_dataset_manifest(root)
    model_df = build_model_manifest(root)
    runtime_df = build_runtime_manifest(root)
    inventory_df = build_artifact_inventory(root)
    manifest_df = build_reproducibility_manifest(root)

    dataset_path = output_dir / "dataset_manifest.csv"
    model_path = output_dir / "model_manifest.csv"
    runtime_path = output_dir / "runtime_manifest.csv"
    inventory_path = output_dir / "artifact_inventory.csv"
    manifest_path = output_dir / "reproducibility_manifest.csv"
    markdown_path = output_dir / "environment_summary.md"

    dataset_df.to_csv(dataset_path, index=False)
    model_df.to_csv(model_path, index=False)
    runtime_df.to_csv(runtime_path, index=False)
    inventory_df.to_csv(inventory_path, index=False)
    manifest_df.to_csv(manifest_path, index=False)

    project_root = root.parent
    markdown = [
        "# Environment Summary",
        "",
        "This appendix-ready export captures the runtime context and the file inventory used to reproduce the benchmark outputs without relying on notebook state.",
        "",
        "## Runtime",
        "",
        _render_markdown_table(runtime_df, ["key", "value"]),
        "",
        "## Dataset Manifest",
        "",
        _render_markdown_table(
            dataset_df,
            [
                "dataset_name",
                "domain",
                "target_column",
                "sensitive_attr",
                "train_rows",
                "test_rows",
                "train_present",
                "test_present",
            ],
        ),
        "",
        "## Model Manifest",
        "",
        _render_markdown_table(
            model_df,
            [
                "model_id",
                "display_name",
                "available",
                "trainable",
                "adult_source_present",
                "adult_source_bytes",
            ],
        ),
        "",
        "## Artifact Inventory",
        "",
        _render_markdown_table(
            inventory_df,
            ["category", "name", "present", "bytes", "mtime_utc"],
        ),
        "",
        "## Scope Notes",
        "",
        f"- Benchmark root: {root}",
        f"- Project root: {project_root}",
        "- Missing optional files are recorded as absent rather than causing export failure.",
    ]
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    return {
        "dataset_manifest": dataset_path,
        "model_manifest": model_path,
        "runtime_manifest": runtime_path,
        "artifact_inventory": inventory_path,
        "reproducibility_manifest": manifest_path,
        "environment_summary": markdown_path,
    }


def main() -> None:
    outputs = write_reproducibility_exports()
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
