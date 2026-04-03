"""
main.py — Runs the complete Principled Synthetic Data Generation Lifecycle.

Conditionally runs stages only if their output files don't exist.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pandas as pd

import config


# Force UTF-8 for this process and all child processes (Windows fix)
os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_script(module_name: str) -> None:
    """Run a module as a script using the current Python executable."""
    cmd = [sys.executable, "-m", module_name]
    subprocess.run(cmd, check=True)


def main() -> None:
    print("=" * 60)
    print("   PRINCIPLED SYNTHETIC DATA GENERATION LIFECYCLE")
    print("   UCI Adult Income Dataset")
    print("=" * 60)

    start_total = time.time()
    adult_train_path = config.DATA_DIR / "adult_train.csv"
    adult_test_path = config.DATA_DIR / "adult_test.csv"
    ctgan_synth_path = config.RESULTS_DIR / "ctgan_synthetic.csv"
    tvae_synth_path = config.RESULTS_DIR / "tvae_synthetic.csv"
    privacy_results_path = config.RESULTS_DIR / "privacy_fairness_results.csv"
    final_eval_path = config.RESULTS_DIR / "final_evaluation_table.csv"
    plot_paths = [
        config.RESULTS_DIR / "plot1_evaluation_dashboard.png",
        config.RESULTS_DIR / "plot2_distribution_comparison.png",
        config.RESULTS_DIR / "plot3_privacy_utility_tradeoff.png",
    ]

    print("\n[STAGE 1] Problem Definition & Data Loading...")
    if adult_train_path.exists() and adult_test_path.exists():
        print("  Data already loaded - skipping")
    else:
        run_script("data.loader")

    print("\n[STAGE 2] Generative Model Training...")
    if ctgan_synth_path.exists() and tvae_synth_path.exists():
        print("  Synthetic data already exists - skipping retraining")
    else:
        run_script("models.train_models")

    print("\n[STAGE 3] Privacy & Fairness Evaluation...")
    if privacy_results_path.exists():
        print("  Privacy results already exist - skipping")
    else:
        run_script("evaluation.privacy_fairness")

    print("\n[STAGE 4] Full Evaluation Dashboard...")
    if final_eval_path.exists():
        print("  Evaluation results already exist - skipping")
    else:
        run_script("evaluation.metrics")

    print("\n[PLOTS] Generating Publication Visualizations...")
    if all(path.exists() for path in plot_paths):
        print("  Plots already exist - skipping")
    else:
        run_script("results.visualize")

    print("\n" + "-" * 60)
    print("FINAL EVALUATION TABLE:")
    try:
        df = pd.read_csv(final_eval_path)
        if "TSTR_Accuracy" in df.columns:
            df["TSTR_Accuracy"] = df["TSTR_Accuracy"].apply(
                lambda value: f"{value:.1f}%"
                if pd.notna(value) and not isinstance(value, str)
                else value
            )
        print(df.to_string(index=False))
    except Exception as exc:
        print(f"  Could not load results table: {exc}")
    print("-" * 60)

    elapsed = time.time() - start_total
    print("\n" + "=" * 60)
    print("   LIFECYCLE COMPLETE")
    print(f"  Total runtime: {elapsed:.1f} seconds")
    print("=" * 60)
    print(f"\n  Results saved to {config.RESULTS_DIR}")
    print("  |- ctgan_synthetic.csv")
    print("  |- tvae_synthetic.csv")
    print("  |- privacy_fairness_results.csv")
    print("  |- final_evaluation_table.csv")
    print("  |- plot1_evaluation_dashboard.png")
    print("  |- plot2_distribution_comparison.png")
    print("  `- plot3_privacy_utility_tradeoff.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
