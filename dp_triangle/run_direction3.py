"""Direction 3 orchestrator for full and dry-run execution."""

from __future__ import annotations

import argparse
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
from dp_triangle import evaluate_triangle, visualize_triangle
from dp_triangle.common import DATASET_SPECS, dataset_input_paths, dataset_result_dir, get_dataset_spec
from dp_triangle.train_dp_variants import model_path as training_model_path
from dp_triangle.train_dp_variants import synthetic_path as training_synthetic_path

torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EXPECTED_TRAIN_SHAPE = (36177, 15)
EXPECTED_TEST_SHAPE = (9045, 15)
VARIANT_TIME_ESTIMATES = {
    "no_dp": {"cuda": 8, "cpu": 20},
    "eps_10": {"cuda": 10, "cpu": 25},
    "eps_1": {"cuda": 12, "cpu": 30},
    "eps_0_5": {"cuda": 15, "cpu": 35},
    "eps_0_1": {"cuda": 20, "cpu": 45},
}
REFRESHABLE_RESULT_FILES = [
    "dp_triangle_dashboard.csv",
    "dp_subgroup_fairness.csv",
    "dp_post_hoc_debiasing.csv",
    "figure4_epsilon_tradeoff_curve.png",
    "figure5_pff_radar_chart.png",
    "figure6_intersectional_fairness.png",
    "figure7_post_hoc_recovery.png",
    "dp_direction3_dry_run.marker",
]


def triangle_score_column(dashboard_df: pd.DataFrame) -> str:
    """
    Choose the preferred triangle score column for reporting.

    Inputs: dashboard dataframe.
    Outputs: preferred triangle score column name.
    Lifecycle stage: Stage 7 - Orchestration.
    Reference: collapse-aware Direction 3 reporting design.
    """
    if "Triangle_Score_Adjusted" in dashboard_df.columns:
        return "Triangle_Score_Adjusted"
    return "Triangle_Score"


def synthetic_path(variant: str, dataset_name: str = "adult") -> Path:
    """
    Return the synthetic CSV path for a dataset variant.

    Inputs: variant key and dataset name.
    Outputs: synthetic CSV path.
    Lifecycle stage: Stage 7 - Orchestration.
    Reference: Direction 3 output specification generalized to supporting datasets.
    """
    return training_synthetic_path(variant, dataset_name=dataset_name)


def model_path(variant: str, dataset_name: str = "adult") -> Path:
    """
    Return the saved model path for a dataset variant.

    Inputs: variant key and dataset name.
    Outputs: saved model pickle path.
    Lifecycle stage: Stage 7 - Orchestration.
    Reference: Direction 3 output specification generalized to supporting datasets.
    """
    return training_model_path(variant, dataset_name=dataset_name)


def dry_run_marker_path(dataset_name: str = "adult") -> Path:
    """
    Return the dry-run marker path for one dataset.

    Inputs: dataset name.
    Outputs: dry-run marker path.
    Lifecycle stage: Stage 7 - Orchestration.
    Reference: Direction 3 dry-run cache invalidation design.
    """
    return dataset_result_dir(dataset_name) / "dp_direction3_dry_run.marker"


def clear_direction3_result_artifacts(dataset_name: str = "adult") -> None:
    """
    Remove cached Direction 3 evaluation and figure artifacts for one dataset.

    Inputs: dataset name.
    Outputs: cached evaluation and figure files removed in place.
    Lifecycle stage: Stage 7 - Orchestration.
    Reference: paper-readiness refresh workflow for dataset-local Direction 3 outputs.
    """
    result_root = dataset_result_dir(dataset_name)
    for name in REFRESHABLE_RESULT_FILES:
        (result_root / name).unlink(missing_ok=True)


def print_environment_banner() -> None:
    """
    Print Python, Torch, Opacus, and CUDA environment details.

    Inputs: none.
    Outputs: environment diagnostics to stdout.
    Lifecycle stage: Stage 7 - Orchestration.
    Reference: Colab execution requirements.
    """
    try:
        import opacus  # type: ignore

        opacus_version = opacus.__version__
    except Exception:
        opacus_version = "not-installed"

    print(f"[Python] {platform.python_version()}")
    print(f"[Torch] {torch.__version__}")
    print(f"[Opacus] {opacus_version}")
    print(f"[CUDA] Available={torch.cuda.is_available()}")


def validate_dataset_inputs(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate train/test inputs and load them for one dataset.

    Inputs: dataset name.
    Outputs: validated train/test dataframes.
    Lifecycle stage: Stage 7 - Orchestration.
    Reference: Direction 3 dry-run and pre-flight requirements generalized to supporting datasets.
    """
    paths = dataset_input_paths(dataset_name)
    train_path = paths["train"]
    test_path = paths["test"]
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path.name} not found. Prepare the dataset splits before running Direction 3 for {dataset_name}."
        )
    if not test_path.exists():
        raise FileNotFoundError(
            f"{test_path.name} not found. Prepare the dataset splits before running Direction 3 for {dataset_name}."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if train_df.empty or test_df.empty:
        raise ValueError(f"{dataset_name} train/test inputs must be non-empty")
    if dataset_name == "adult":
        if train_df.shape != EXPECTED_TRAIN_SHAPE:
            raise ValueError(f"adult_train.csv has expected shape {EXPECTED_TRAIN_SHAPE}, found {train_df.shape}")
        if test_df.shape != EXPECTED_TEST_SHAPE:
            raise ValueError(f"adult_test.csv has expected shape {EXPECTED_TEST_SHAPE}, found {test_df.shape}")
    return train_df, test_df


def generate_dummy_synthetic_csvs(train_df: pd.DataFrame, dataset_name: str = "adult") -> None:
    """
    Generate dummy synthetic CSVs for dry-run validation.

    Inputs: real train dataframe and dataset name.
    Outputs: one dummy synthetic CSV per epsilon variant.
    Lifecycle stage: Stage 7 - Dry-run validation.
    Reference: Direction 3 dry-run specification generalized to supporting datasets.
    """
    result_root = dataset_result_dir(dataset_name)
    result_root.mkdir(parents=True, exist_ok=True)
    dry_run_marker_path(dataset_name).write_text("dry-run outputs present\n", encoding="utf-8")
    for key in config.DP_EPSILON_VALUES:
        sampled = train_df.sample(
            n=config.DP_N_SYNTHETIC,
            replace=True,
            random_state=config.RANDOM_SEED,
        ).reset_index(drop=True)
        sampled.to_csv(synthetic_path(key, dataset_name=dataset_name), index=False)
        print(f"[DRY-RUN] Generated dummy synthetic data for {key}")


def estimate_training_minutes(pending_variants: list[str]) -> int:
    """
    Estimate training time for the remaining variants.

    Inputs: uncached variant keys.
    Outputs: estimated total minutes for the current device.
    Lifecycle stage: Stage 7 - Orchestration.
    Reference: Direction 3 runtime guidance.
    """
    device_key = "cuda" if torch.cuda.is_available() else "cpu"
    return int(sum(VARIANT_TIME_ESTIMATES[variant][device_key] for variant in pending_variants))


def _format_metric(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.3f}"


def print_final_summary(dataset_name: str = "adult") -> None:
    """
    Print the final Direction 3 summary tables for one dataset.

    Inputs: dataset name.
    Outputs: formatted stdout summary.
    Lifecycle stage: Stage 7 - Orchestration.
    Reference: Direction 3 final summary specification generalized to supporting datasets.
    """
    spec = get_dataset_spec(dataset_name)
    result_root = dataset_result_dir(dataset_name)
    dashboard_df = pd.read_csv(result_root / "dp_triangle_dashboard.csv")
    score_column = triangle_score_column(dashboard_df)

    print("=" * 68)
    print(f" DIRECTION 3 RESULTS - {dataset_name.upper()}")
    print("=" * 68)
    print("Variant        TSTR%    JS      MIA_Adv   Demo_Par   Tri.Score")
    print("-" * 68)
    for variant in ["no_dp", "eps_10", "eps_1", "eps_0_5", "eps_0_1"]:
        row = dashboard_df.loc[dashboard_df["variant"] == variant].iloc[0]
        marker = " *" if bool(row.get("Collapsed_Minority_Class", False)) else ""
        print(
            f"{visualize_triangle.VARIANT_LABELS[variant] + marker:<13}"
            f"{row['TSTR']:>7.2f}%  "
            f"{_format_metric(row['JS']):>6}  "
            f"{_format_metric(row['MIA_Advantage']):>8}  "
            f"{_format_metric(row['Demo_Parity']):>8}  "
            f"{_format_metric(row[score_column]):>9}"
        )

    subgroup_path = result_root / "dp_subgroup_fairness.csv"
    if spec.supports_full_triangle and subgroup_path.exists():
        subgroup_df = pd.read_csv(subgroup_path).set_index("variant")
        print()
        print("Subgroup Utility:")
        print("-" * 68)
        subgroup_label = dashboard_df["Sensitive_Subgroup_Label"].dropna().iloc[0] if dashboard_df["Sensitive_Subgroup_Label"].notna().any() else "sensitive"
        for variant in ["no_dp", "eps_0_1"]:
            if variant not in subgroup_df.index:
                continue
            row = subgroup_df.loc[variant]
            print(
                f"{visualize_triangle.VARIANT_LABELS[variant]:<13}"
                f"overall={row['tstr_overall']:.2f}  "
                f"{subgroup_label}={row['tstr_sensitive_subgroup']:.2f} "
                f"({row['sensitive_subgroup_degradation']:.2f}%)  "
                f"positive={row['tstr_positive_class']:.2f} "
                f"({row['positive_class_degradation']:.2f}%)"
            )

    posthoc_path = result_root / "dp_post_hoc_debiasing.csv"
    if posthoc_path.exists():
        posthoc_df = pd.read_csv(posthoc_path)
        successful = posthoc_df.loc[posthoc_df["debiasing_status"] == "success"]
        if not successful.empty:
            best_recovery = successful.loc[successful["fairness_recovery_rate_%"].idxmax()]
            print()
            print(
                f"Best fairness recovery: {best_recovery['epsilon_label']} - "
                f"{best_recovery['fairness_recovery_rate_%']:.2f}% at "
                f"{best_recovery['utility_cost']:.2f}% utility cost"
            )

    collapsed_mask = pd.Series(False, index=dashboard_df.index)
    if "Collapsed_Minority_Class" in dashboard_df.columns:
        collapsed_mask = dashboard_df["Collapsed_Minority_Class"].fillna(False).astype(bool)
    collapsed = dashboard_df.loc[collapsed_mask]
    if not collapsed.empty:
        collapse_notes = ", ".join(
            f"{row['epsilon_label']} ({row['Collapse_Reason']})"
            for _, row in collapsed.iterrows()
        )
        print(f"Collapsed variants: {collapse_notes}")

    if dashboard_df[score_column].notna().any():
        best_triangle = dashboard_df.loc[dashboard_df[score_column].idxmax()]
        print(f"Best Triangle Score: {best_triangle['epsilon_label']} with score {best_triangle[score_column]:.3f}")
    print(f"Outputs saved to: {result_root}")
    print("=" * 68)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """
    Parse CLI arguments.

    Inputs: optional argv list.
    Outputs: parsed CLI namespace.
    Lifecycle stage: Stage 7 - Orchestration.
    Reference: standard argparse usage.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="adult")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh-results", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Execute the Direction 3 pipeline for one dataset.

    Inputs: optional CLI argv.
    Outputs: integer exit code.
    Lifecycle stage: Stage 7 - Orchestration.
    Reference: Direction 3 pipeline specification generalized to supporting datasets.
    """
    args = parse_args(argv)
    dataset_name = args.dataset
    spec = get_dataset_spec(dataset_name)

    dataset_result_dir(dataset_name).mkdir(parents=True, exist_ok=True)
    if dataset_name == "adult":
        config.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    train_df, _ = validate_dataset_inputs(dataset_name)

    if args.dry_run:
        generate_dummy_synthetic_csvs(train_df, dataset_name=dataset_name)
        evaluate_triangle.main([dataset_name])
        visualize_triangle.main([dataset_name])
        print(
            "[DRY-RUN COMPLETE] All pipeline stages validated. "
            "No models were trained. Replace dummy CSVs with real outputs before interpreting results."
        )
        return 0

    if args.refresh_results:
        clear_direction3_result_artifacts(dataset_name)

    print_environment_banner()
    pending = [
        variant
        for variant in config.DP_EPSILON_VALUES
        if not (
            model_path(variant, dataset_name=dataset_name).exists()
            and synthetic_path(variant, dataset_name=dataset_name).exists()
        )
    ]
    print(f"Training {len(pending)}/5 variants (others cached)")
    print(f"Estimated total time: ~{estimate_training_minutes(pending)} minutes")

    from dp_triangle import post_hoc_debiasing, train_dp_variants

    train_dp_variants.main([dataset_name])
    evaluate_triangle.main([dataset_name])
    if spec.supports_full_triangle:
        post_hoc_debiasing.main([dataset_name])
    visualize_triangle.main([dataset_name])
    if dry_run_marker_path(dataset_name).exists():
        dry_run_marker_path(dataset_name).unlink()
    print_final_summary(dataset_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
