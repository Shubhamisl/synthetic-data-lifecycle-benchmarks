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
REFERENCE_ROWS = [
    ("CTGAN (ref)", "81.43%", "0.053", "0.490", "0.220", "N/A"),
    ("TVAE  (ref)", "82.19%", "0.035", "0.376", "0.181", "N/A"),
    ("Real  (ref)", "85.42%", "N/A", "0.000", "0.200", "N/A"),
]


def triangle_score_column(dashboard_df: pd.DataFrame) -> str:
    """
    Choose the preferred triangle score column for reporting.

    Inputs: dashboard dataframe.
    Outputs: column name for the preferred triangle score.
    Lifecycle stage: Stage 7 - Orchestration.
    Reference: collapse-aware Direction 3 reporting design.
    """
    if "Triangle_Score_Adjusted" in dashboard_df.columns:
        return "Triangle_Score_Adjusted"
    return "Triangle_Score"


def synthetic_path(variant: str) -> Path:
    """
    Return the synthetic CSV path for a variant.

    Inputs: variant key.
    Outputs: synthetic CSV path.
    Lifecycle stage: Stage 7 - Orchestration.
    Reference: Direction 3 output specification.
    """
    return config.RESULTS_DIR / f"dp_synthetic_{variant}.csv"


def model_path(variant: str) -> Path:
    """
    Return the saved model path for a variant.

    Inputs: variant key.
    Outputs: saved model pickle path.
    Lifecycle stage: Stage 7 - Orchestration.
    Reference: Direction 3 output specification.
    """
    return config.MODEL_SAVE_DIR / f"dp_ctgan_{variant}.pkl"


def dry_run_marker_path() -> Path:
    """
    Return the dry-run marker path.

    Inputs: none.
    Outputs: dry-run marker path.
    Lifecycle stage: Stage 7 - Orchestration.
    Reference: Direction 3 dry-run cache invalidation design.
    """
    return config.RESULTS_DIR / "dp_direction3_dry_run.marker"


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


def validate_adult_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate Adult train/test inputs and load them.

    Inputs: none.
    Outputs: validated Adult train/test dataframes.
    Lifecycle stage: Stage 7 - Orchestration.
    Reference: Direction 3 dry-run and pre-flight requirements.
    """
    train_path = config.DATA_DIR / "adult_train.csv"
    test_path = config.DATA_DIR / "adult_test.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            "adult_train.csv not found. Run 'python main.py' first to download and split the UCI Adult dataset."
        )
    if not test_path.exists():
        raise FileNotFoundError(
            "adult_test.csv not found. Run 'python main.py' first to download and split the UCI Adult dataset."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if train_df.shape != EXPECTED_TRAIN_SHAPE:
        raise ValueError(f"adult_train.csv has expected shape {EXPECTED_TRAIN_SHAPE}, found {train_df.shape}")
    if test_df.shape != EXPECTED_TEST_SHAPE:
        raise ValueError(f"adult_test.csv has expected shape {EXPECTED_TEST_SHAPE}, found {test_df.shape}")
    return train_df, test_df


def generate_dummy_synthetic_csvs(train_df: pd.DataFrame) -> None:
    """
    Generate dummy synthetic CSVs for dry-run validation.

    Inputs: real Adult train dataframe.
    Outputs: one dummy synthetic CSV per epsilon variant.
    Lifecycle stage: Stage 7 - Dry-run validation.
    Reference: Direction 3 dry-run specification.
    """
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dry_run_marker_path().write_text("dry-run outputs present\n", encoding="utf-8")
    for key in config.DP_EPSILON_VALUES:
        sampled = train_df.sample(
            n=config.DP_N_SYNTHETIC,
            replace=True,
            random_state=config.RANDOM_SEED,
        ).reset_index(drop=True)
        sampled.to_csv(synthetic_path(key), index=False)
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


def print_final_summary() -> None:
    """
    Print the final Direction 3 summary tables.

    Inputs: none.
    Outputs: formatted stdout summary table.
    Lifecycle stage: Stage 7 - Orchestration.
    Reference: Direction 3 final summary specification.
    """
    dashboard_df = pd.read_csv(config.RESULTS_DIR / "dp_triangle_dashboard.csv")
    subgroup_df = pd.read_csv(config.RESULTS_DIR / "dp_subgroup_fairness.csv").set_index("variant")
    score_column = triangle_score_column(dashboard_df)

    print("═" * 68)
    print(" DIRECTION 3 RESULTS — Privacy–Fairness–Fidelity Triangle")
    print("═" * 68)
    print("┌─────────────┬────────┬───────┬──────────┬────────────┬──────────────┐")
    print("│ Variant     │  TSTR% │  JS↓  │  MIA_Adv↓│ Demo_Par↓  │ Tri. Score↑ │")
    print("├─────────────┼────────┼───────┼──────────┼────────────┼──────────────┤")
    for variant in ["no_dp", "eps_10", "eps_1", "eps_0_5", "eps_0_1"]:
        row = dashboard_df.loc[dashboard_df["variant"] == variant].iloc[0]
        marker = " *" if bool(row.get("Collapsed_Minority_Class", False)) else ""
        label = f"{visualize_triangle.VARIANT_LABELS[variant]}{marker}"
        print(
            f"│ {label:<11} │ "
            f"{row['TSTR']:>6.2f}% │ {row['JS']:>5.3f} │ "
            f"{row['MIA_Advantage']:>8.3f} │ {row['Demo_Parity']:>10.3f} │ "
            f"{row[score_column]:>10.3f} │"
        )
    print("├─────────────┼────────┼───────┼──────────┼────────────┼──────────────┤")
    for label, tstr, js_value, mia_value, dp_value, triangle in REFERENCE_ROWS:
        print(f"│ {label:<11} │ {tstr:>6} │ {js_value:>5} │ {mia_value:>8} │ {dp_value:>10} │ {triangle:>10} │")
    print("└─────────────┴────────┴───────┴──────────┴────────────┴──────────────┘")
    print()
    print("Intersectional Fairness (subgroup degradation under DP):")
    print("┌─────────────┬──────────────┬──────────────────────┬──────────────────────┐")
    print("│ Variant     │ Overall TSTR │ Female TSTR (Δ)       │ >50K TSTR (Δ)        │")
    print("├─────────────┼──────────────┼──────────────────────┼──────────────────────┤")
    for variant in ["no_dp", "eps_0_1"]:
        row = subgroup_df.loc[variant]
        print(
            f"│ {visualize_triangle.VARIANT_LABELS[variant]:<11} │ "
            f"{row['tstr_overall']:>11.2f} │ {row['tstr_female']:>6.2f} ({row['female_degradation']:.2f}%)"
            f"{' ' * 8}│ {row['tstr_high_income']:>6.2f} ({row['high_income_degradation']:.2f}%)"
            f"{' ' * 7}│"
        )
    print("└─────────────┴──────────────┴──────────────────────┴──────────────────────┘")

    posthoc_path = config.RESULTS_DIR / "dp_post_hoc_debiasing.csv"
    if posthoc_path.exists():
        posthoc_df = pd.read_csv(posthoc_path)
        successful = posthoc_df.loc[posthoc_df["debiasing_status"] == "success"]
        if not successful.empty:
            best_recovery = successful.loc[successful["fairness_recovery_rate_%"].idxmax()]
            print(
                f"Post-Hoc Debiasing Summary:\n"
                f"  Best fairness recovery: {best_recovery['epsilon_label']} — "
                f"{best_recovery['fairness_recovery_rate_%']:.2f}% recovered at "
                f"{best_recovery['utility_cost']:.2f}% TSTR cost"
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
        print(f"  Collapsed variants:      {collapse_notes}")

    best_triangle = dashboard_df.loc[dashboard_df[score_column].idxmax()]
    print(
        f"  Best Triangle Score:    {best_triangle['epsilon_label']} "
        f"with score {best_triangle[score_column]:.3f}"
    )
    print(f"\nOutputs saved to: {config.RESULTS_DIR}")
    print("═" * 68)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """
    Parse CLI arguments.

    Inputs: optional argv list.
    Outputs: parsed CLI namespace.
    Lifecycle stage: Stage 7 - Orchestration.
    Reference: standard argparse usage.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Execute the Direction 3 pipeline.

    Inputs: optional CLI argv.
    Outputs: integer exit code.
    Lifecycle stage: Stage 7 - Orchestration.
    Reference: Direction 3 pipeline specification.
    """
    args = parse_args(argv)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    train_df, _ = validate_adult_inputs()

    if args.dry_run:
        generate_dummy_synthetic_csvs(train_df)
        evaluate_triangle.main([])
        visualize_triangle.main([])
        print(
            "[DRY-RUN COMPLETE] All pipeline stages validated. "
            "No models were trained. Replace dummy CSVs with real outputs before interpreting results."
        )
        return 0

    print_environment_banner()
    pending = [
        variant
        for variant in config.DP_EPSILON_VALUES
        if not (model_path(variant).exists() and synthetic_path(variant).exists())
    ]
    print(f"Training {len(pending)}/5 variants (others cached)")
    print(f"Estimated total time: ~{estimate_training_minutes(pending)} minutes")

    from dp_triangle import post_hoc_debiasing, train_dp_variants

    train_dp_variants.main([])
    evaluate_triangle.main([])
    post_hoc_debiasing.main([])
    visualize_triangle.main([])
    if dry_run_marker_path().exists():
        dry_run_marker_path().unlink()
    print_final_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
