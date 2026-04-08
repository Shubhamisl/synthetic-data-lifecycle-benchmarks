from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .benchmark_models import get_benchmark_models
from .common import BENCHMARK_ROOT, DATASET_REGISTRY

RESULTS_DIR = BENCHMARK_ROOT / "results" / "compute"
OUTPUT_CSV = RESULTS_DIR / "compute_summary.csv"
OUTPUT_MD = RESULTS_DIR / "compute_summary.md"


def _utc_timestamp(path: Path | None) -> str:
    if path is None or not path.exists():
        return "N/A"
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _path_stats(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {
            "path": "N/A",
            "present": False,
            "bytes": None,
            "mtime_utc": "N/A",
        }

    return {
        "path": str(path),
        "present": True,
        "bytes": path.stat().st_size,
        "mtime_utc": _utc_timestamp(path),
    }


def _read_csv_shape(path: Path | None) -> tuple[int | None, int | None]:
    if path is None or not path.exists():
        return None, None

    try:
        dataframe = pd.read_csv(path)
    except Exception:
        return None, None

    return int(dataframe.shape[0]), int(dataframe.shape[1])


def _first_existing_path(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _candidate_model_artifact_paths(benchmark_root: Path, dataset_name: str, model_id: str) -> list[Path]:
    project_root = benchmark_root.parent
    return [
        benchmark_root / "models" / f"{dataset_name}_{model_id}.pkl",
        benchmark_root / "models" / f"{model_id}.pkl",
        benchmark_root / "results" / f"{dataset_name}_{model_id}.pkl",
        project_root / "colab_artifacts" / "models" / "saved" / f"{model_id}.pkl",
        project_root / "colab_artifacts" / "benchmarks" / "results" / "dp_triangle" / dataset_name / "models" / f"{model_id}.pkl",
    ]


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


def build_compute_summary(benchmark_root: Path | None = None) -> pd.DataFrame:
    root = benchmark_root or BENCHMARK_ROOT
    rows: list[dict[str, object]] = []

    benchmark_log = root / "results" / "train_benchmark_models_output.log"
    benchmark_log_stats = _path_stats(benchmark_log)

    for dataset_name, spec in DATASET_REGISTRY.items():
        train_path = root / "datasets" / f"{dataset_name}_train.csv"
        test_path = root / "datasets" / f"{dataset_name}_test.csv"
        dataset_rows, dataset_features = _read_csv_shape(train_path)
        test_rows, test_features = _read_csv_shape(test_path)

        for model_spec in get_benchmark_models():
            synthetic_path = root / "synthetic" / f"{dataset_name}_{model_spec.model_id}.csv"
            synthetic_rows, synthetic_features = _read_csv_shape(synthetic_path)
            synthetic_stats = _path_stats(synthetic_path)

            model_artifact_path = _first_existing_path(
                _candidate_model_artifact_paths(root, dataset_name, model_spec.model_id)
            )
            model_stats = _path_stats(model_artifact_path)

            notes: list[str] = []
            if not synthetic_stats["present"]:
                notes.append("synthetic output missing")
            if not model_stats["present"]:
                notes.append("model artifact unavailable")
            if dataset_rows is None:
                notes.append("dataset train file missing")

            rows.append(
                {
                    "dataset_name": dataset_name,
                    "dataset_domain": spec.domain,
                    "row_count": dataset_rows,
                    "feature_count": dataset_features,
                    "test_row_count": test_rows,
                    "test_feature_count": test_features,
                    "model_id": model_spec.model_id,
                    "model_display_name": model_spec.display_name,
                    "synthetic_output_present": synthetic_stats["present"],
                    "synthetic_output_path": synthetic_stats["path"],
                    "synthetic_output_rows": synthetic_rows,
                    "synthetic_output_features": synthetic_features,
                    "synthetic_output_bytes": synthetic_stats["bytes"],
                    "synthetic_output_mtime_utc": synthetic_stats["mtime_utc"],
                    "model_artifact_present": model_stats["present"],
                    "model_artifact_path": model_stats["path"],
                    "model_artifact_bytes": model_stats["bytes"],
                    "model_artifact_mtime_utc": model_stats["mtime_utc"],
                    "runtime_log_present": benchmark_log_stats["present"],
                    "runtime_log_path": benchmark_log_stats["path"],
                    "runtime_log_bytes": benchmark_log_stats["bytes"],
                    "runtime_log_mtime_utc": benchmark_log_stats["mtime_utc"],
                    "notes": "; ".join(notes) if notes else "ok",
                }
            )

    return pd.DataFrame(rows)


def write_compute_summary(benchmark_root: Path | None = None) -> dict[str, Path]:
    root = benchmark_root or BENCHMARK_ROOT
    output_dir = root / "results" / "compute"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = build_compute_summary(root)
    summary_df.to_csv(output_dir / "compute_summary.csv", index=False)

    if not summary_df.empty:
        display_columns = [
            "dataset_name",
            "model_display_name",
            "row_count",
            "feature_count",
            "synthetic_output_present",
            "synthetic_output_bytes",
            "model_artifact_present",
            "model_artifact_bytes",
            "notes",
        ]
        total_rows = int(summary_df["synthetic_output_present"].sum())
        markdown = [
            "# Compute Summary",
            "",
            f"Generated on {datetime.now(timezone.utc).isoformat()}",
            "",
            f"- Datasets covered: {summary_df['dataset_name'].nunique()}",
            f"- Model entries: {len(summary_df)}",
            f"- Synthetic outputs present: {total_rows}",
            "",
            "## Appendix-Ready Inventory",
            "",
            _render_markdown_table(summary_df, display_columns),
        ]
    else:
        markdown = [
            "# Compute Summary",
            "",
            "No compute artifacts were discovered.",
        ]

    (output_dir / "compute_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return {
        "csv": output_dir / "compute_summary.csv",
        "markdown": output_dir / "compute_summary.md",
    }


def main() -> None:
    outputs = write_compute_summary()
    print(f"Compute summary saved to {outputs['csv']}")
    print(f"Compute markdown saved to {outputs['markdown']}")


if __name__ == "__main__":
    main()
