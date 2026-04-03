"""
main.py — Runs the complete Principled Synthetic Data Generation Lifecycle.

Conditionally runs stages only if their output files don't exist.
"""

import os
import time
import pandas as pd
import subprocess
import sys

# Force UTF-8 for this process and all child processes (Windows fix)
os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_script(module_name: str):
    """Run a module as a script using the current Python executable."""
    cmd = [sys.executable, "-m", module_name]
    subprocess.run(cmd, check=True)


def main():
    print("=" * 60)
    print("   PRINCIPLED SYNTHETIC DATA GENERATION LIFECYCLE")
    print("   UCI Adult Income Dataset")
    print("=" * 60)

    start_total = time.time()

    # ── STAGE 1 ──────────────────────────────────────────────────────
    print("\n[STAGE 1] Problem Definition & Data Loading...")
    if os.path.exists("data/adult_train.csv") and os.path.exists("data/adult_test.csv"):
        print("  ✓ Data already loaded — skipping")
    else:
        run_script("data.loader")

    # ── STAGE 2 ──────────────────────────────────────────────────────
    print("\n[STAGE 2] Generative Model Training...")
    if os.path.exists("results/ctgan_synthetic.csv") and os.path.exists("results/tvae_synthetic.csv"):
        print("  ✓ Synthetic data already exists — skipping retraining")
    else:
        run_script("models.train_models")

    # ── STAGE 3 ──────────────────────────────────────────────────────
    print("\n[STAGE 3] Privacy & Fairness Evaluation...")
    if os.path.exists("results/privacy_fairness_results.csv"):
        print("  ✓ Privacy results already exist — skipping")
    else:
        run_script("evaluation.privacy_fairness")

    # ── STAGE 4 ──────────────────────────────────────────────────────
    print("\n[STAGE 4] Full Evaluation Dashboard...")
    if os.path.exists("results/final_evaluation_table.csv"):
        print("  ✓ Evaluation results already exist — skipping")
    else:
        run_script("evaluation.metrics")

    # ── VISUALIZATIONS ───────────────────────────────────────────────
    print("\n[PLOTS] Generating Publication Visualizations...")
    plots = [
        "results/plot1_evaluation_dashboard.png",
        "results/plot2_distribution_comparison.png",
        "results/plot3_privacy_utility_tradeoff.png",
    ]
    if all(os.path.exists(p) for p in plots):
        print("  ✓ Plots already exist — skipping")
    else:
        run_script("results.visualize")

    # ── FINAL SUMMARY ────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("FINAL EVALUATION TABLE:")
    try:
        df = pd.read_csv("results/final_evaluation_table.csv")
        # Format the TSTR Accuracy if it's numeric
        if 'TSTR_Accuracy' in df.columns:
            df['TSTR_Accuracy'] = df['TSTR_Accuracy'].apply(
                lambda x: f"{x:.1f}%" if pd.notna(x) and not isinstance(x, str) else x
            )
        print(df.to_string(index=False))
    except Exception as e:
        print(f"  Could not load results table: {e}")
    print("-" * 60)

    elapsed = time.time() - start_total
    print("\n" + "=" * 60)
    print("   LIFECYCLE COMPLETE")
    print(f"  Total runtime: {elapsed:.1f} seconds")
    print("=" * 60)
    print("\n  Results saved to results/")
    print("  ├── ctgan_synthetic.csv")
    print("  ├── tvae_synthetic.csv")
    print("  ├── privacy_fairness_results.csv")
    print("  ├── final_evaluation_table.csv")
    print("  ├── plot1_evaluation_dashboard.png")
    print("  ├── plot2_distribution_comparison.png")
    print("  └── plot3_privacy_utility_tradeoff.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
