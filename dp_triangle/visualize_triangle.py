"""Direction 3 plotting for privacy-fairness-fidelity results."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config

torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)

VARIANT_ORDER_PRIVACY = ["eps_0_1", "eps_0_5", "eps_1", "eps_10", "no_dp"]
VARIANT_ORDER_STANDARD = ["no_dp", "eps_10", "eps_1", "eps_0_5", "eps_0_1"]
VARIANT_LABELS = {
    "no_dp": "No DP",
    "eps_10": "ε=10.0",
    "eps_1": "ε=1.0",
    "eps_0_5": "ε=0.5",
    "eps_0_1": "ε=0.1",
}
COLORS = {
    "no_dp": "#FF7F0E",
    "eps_10": "#FFBF00",
    "eps_1": "#2CA02C",
    "eps_0_5": "#17BECF",
    "eps_0_1": "#1F77B4",
    "Real": "#000000",
    "CTGAN": "#D62728",
    "TVAE": "#9467BD",
}


def result_paths() -> dict[str, Path]:
    """Inputs: none. Outputs: figure and CSV paths for Direction 3 plotting. Lifecycle stage: Stage 6 — Visualisation. Reference: Direction 3 plotting specification."""
    return {
        "dashboard": config.RESULTS_DIR / "dp_triangle_dashboard.csv",
        "posthoc": config.RESULTS_DIR / "dp_post_hoc_debiasing.csv",
        "fig4": config.RESULTS_DIR / "figure4_epsilon_tradeoff_curve.png",
        "fig5": config.RESULTS_DIR / "figure5_pff_radar_chart.png",
        "fig6": config.RESULTS_DIR / "figure6_intersectional_fairness.png",
        "fig7": config.RESULTS_DIR / "figure7_post_hoc_recovery.png",
        "baseline": config.RESULTS_DIR / "final_evaluation_table.csv",
        "dry_run_marker": config.RESULTS_DIR / "dp_direction3_dry_run.marker",
    }


def load_reference_rows() -> dict[str, dict[str, float]]:
    """Inputs: none. Outputs: baseline metric mapping for Real, CTGAN, and TVAE. Lifecycle stage: Stage 6 — Visualisation. Reference: original project final evaluation table."""
    baseline_path = result_paths()["baseline"]
    if not baseline_path.exists():
        return {
            "CTGAN": {
                "Model": "CTGAN",
                "JS_Divergence": 0.053,
                "TSTR_Accuracy": 81.43,
                "MIA_Advantage": 0.490,
                "Demographic_Parity": 0.220,
            },
            "TVAE": {
                "Model": "TVAE",
                "JS_Divergence": 0.035,
                "TSTR_Accuracy": 82.19,
                "MIA_Advantage": 0.376,
                "Demographic_Parity": 0.181,
            },
            "Real": {
                "Model": "Real",
                "JS_Divergence": np.nan,
                "TSTR_Accuracy": 85.42,
                "MIA_Advantage": 0.0,
                "Demographic_Parity": 0.200,
            },
        }

    baseline_df = pd.read_csv(baseline_path)
    refs: dict[str, dict[str, float]] = {}
    for _, row in baseline_df.iterrows():
        refs[str(row["Model"])] = row.to_dict()
    return refs


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Inputs: matplotlib figure and output path. Outputs: saved PNG on disk. Lifecycle stage: Stage 6 — Visualisation. Reference: project plotting conventions."""
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_figure4(dashboard_df: pd.DataFrame, refs: dict[str, dict[str, float]]) -> None:
    """Inputs: dashboard dataframe and baseline reference rows. Outputs: epsilon trade-off curve PNG. Lifecycle stage: Stage 6 — Visualisation. Reference: Direction 3 Figure 4 specification."""
    path = result_paths()["fig4"]
    if path.exists() and not result_paths()["dry_run_marker"].exists():
        print(f"[CACHE HIT] Skipping {path.name}")
        return

    ordered = dashboard_df.set_index("variant").loc[VARIANT_ORDER_PRIVACY].reset_index()
    x = np.arange(len(ordered))

    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()

    ax1.plot(x, ordered["TSTR"], color="#1F77B4", marker="o", linewidth=2, label="TSTR Accuracy")
    ax1.axhline(refs["Real"]["TSTR_Accuracy"], color="#1F77B4", linestyle="--", alpha=0.8, label="Real Baseline")
    ax1.set_ylabel("TSTR Accuracy (%)", color="#1F77B4")
    ax1.tick_params(axis="y", labelcolor="#1F77B4")

    ax2.plot(x, ordered["MIA_Advantage"], color="#D62728", marker="^", linewidth=2, label="MIA Advantage")
    ax2.plot(x, ordered["Demo_Parity"], color="#2CA02C", marker="s", linewidth=2, label="Demo. Parity")
    ax2.axhline(0.200, color="#2CA02C", linestyle="--", alpha=0.8, label="Real Demo_Parity")
    ax2.set_ylabel("Metric Value (0-1)")

    ax1.axvspan(-0.5, 1.5, color="#D9D9D9", alpha=0.4)
    ax1.text(0.5, ax1.get_ylim()[1] * 0.98, "GDPR-grade privacy zone", ha="center", va="top")

    ax1.set_xticks(x)
    ax1.set_xticklabels([VARIANT_LABELS[variant] for variant in ordered["variant"]])
    ax1.set_xlabel("Privacy Budget ε (← more private | less private →)")
    ax1.set_title("Privacy Budget ε vs Utility, Privacy Risk, and Fairness")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", bbox_to_anchor=(1.05, 1.0))
    save_figure(fig, path)


def plot_figure5(dashboard_df: pd.DataFrame) -> None:
    """Inputs: dashboard dataframe. Outputs: radar chart PNG. Lifecycle stage: Stage 6 — Visualisation. Reference: Direction 3 Figure 5 specification."""
    path = result_paths()["fig5"]
    if path.exists() and not result_paths()["dry_run_marker"].exists():
        print(f"[CACHE HIT] Skipping {path.name}")
        return

    categories = ["Privacy\n(1−MIA)", "Utility\n(TSTR/100)", "Fairness\n(1−DP)"]
    total_axes = 3
    angles = [index / float(total_axes) * 2 * np.pi for index in range(total_axes)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=13)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_tick_params(labelsize=9)

    for variant in VARIANT_ORDER_STANDARD:
        row = dashboard_df.loc[dashboard_df["variant"] == variant].iloc[0]
        values = [row["Privacy_Score"], row["Utility_Score"], row["Fairness_Score"]]
        values += values[:1]
        label = f"{VARIANT_LABELS[variant]} (Δ={row['Triangle_Score']:.2f})"
        ax.plot(angles, values, color=COLORS[variant], linewidth=2, label=label)
        ax.fill(angles, values, color=COLORS[variant], alpha=0.15)

    ideal_values = [1.0, 1.0, 1.0, 1.0]
    ax.plot(angles, ideal_values, color=COLORS["Real"], linewidth=2, linestyle="--", label="Ideal (score=1.00)")
    ax.set_title("Privacy–Fairness–Fidelity Triangle\nEach polygon = one DP variant. Ideal = all three axes = 1.0", pad=20)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05))
    save_figure(fig, path)


def plot_figure6(dashboard_df: pd.DataFrame, refs: dict[str, dict[str, float]]) -> None:
    """Inputs: dashboard dataframe and baseline reference rows. Outputs: intersectional fairness bar chart PNG. Lifecycle stage: Stage 6 — Visualisation. Reference: Direction 3 Figure 6 specification."""
    path = result_paths()["fig6"]
    if path.exists() and not result_paths()["dry_run_marker"].exists():
        print(f"[CACHE HIT] Skipping {path.name}")
        return

    ordered = dashboard_df.set_index("variant").loc[VARIANT_ORDER_STANDARD].reset_index()
    x = np.arange(len(ordered))
    width = 0.24

    fig, ax = plt.subplots(figsize=(12, 8))
    overall_bars = []
    female_bars = []
    income_bars = []

    for index, row in ordered.iterrows():
        color = COLORS[row["variant"]]
        overall_bars.append(ax.bar(index - width, row["TSTR"], width=width, color=color))
        female_bars.append(ax.bar(index, row["TSTR_Female"], width=width, color=color, hatch="//"))
        income_bars.append(ax.bar(index + width, row["TSTR_High_Income"], width=width, color=color, hatch="xx"))

        female_color = "red" if row["Female_Degradation"] > 0 else "green"
        high_color = "red" if row["High_Income_Degradation"] > 0 else "green"
        ax.text(index, row["TSTR_Female"] + 0.6, f"−{row['Female_Degradation']:.1f}%" if row["Female_Degradation"] > 0 else f"+{abs(row['Female_Degradation']):.1f}%", color=female_color, ha="center", fontweight="bold")
        ax.text(index + width, row["TSTR_High_Income"] + 0.6, f"−{row['High_Income_Degradation']:.1f}%" if row["High_Income_Degradation"] > 0 else f"+{abs(row['High_Income_Degradation']):.1f}%", color=high_color, ha="center", fontweight="bold")

    ax.axhline(refs["Real"]["TSTR_Accuracy"], color=COLORS["Real"], linestyle="--", label="Real Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[variant] for variant in ordered["variant"]])
    ax.set_xlabel("Privacy Budget ε (← more private | less private →)")
    ax.set_ylabel("TSTR Accuracy (%)")
    ax.set_title("Intersectional Fairness: Who Is Most Harmed by DP?")

    legend_handles = [
        overall_bars[0][0],
        female_bars[0][0],
        income_bars[0][0],
    ]
    ax.legend(legend_handles, ["Overall TSTR", "Female Subgroup TSTR", ">50K Income TSTR"])
    save_figure(fig, path)


def plot_figure7(posthoc_df: pd.DataFrame, refs: dict[str, dict[str, float]]) -> None:
    """Inputs: post-hoc debiasing dataframe and baseline references. Outputs: recovery plot PNG. Lifecycle stage: Stage 6 — Visualisation. Reference: Direction 3 Figure 7 specification."""
    path = result_paths()["fig7"]
    if path.exists() and not result_paths()["dry_run_marker"].exists():
        print(f"[CACHE HIT] Skipping {path.name}")
        return

    x = np.arange(len(VARIANT_ORDER_STANDARD))
    ordered = posthoc_df.set_index("variant").loc[VARIANT_ORDER_STANDARD].reset_index()
    valid_mask = ordered["debiasing_status"].eq("success")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    ax1.plot(x[valid_mask], ordered.loc[valid_mask, "dp_before"], color="#333333", linewidth=2, label="Before")
    ax1.plot(x[valid_mask], ordered.loc[valid_mask, "dp_after"], color="#333333", linewidth=2, linestyle="--", label="After")
    ax1.fill_between(x[valid_mask], ordered.loc[valid_mask, "dp_before"], ordered.loc[valid_mask, "dp_after"], color="#90EE90", alpha=0.4, label="Fairness Recovered")
    ax1.axhline(0.200, color=COLORS["Real"], linestyle="--", label="Real Demo_Parity")
    ax1.set_title("Demographic Parity: Before vs After Debiasing")
    ax1.set_ylabel("Demographic Parity Difference (↓ better)")

    ax2.plot(x[valid_mask], ordered.loc[valid_mask, "tstr_before"], color="#333333", linewidth=2, label="Before")
    ax2.plot(x[valid_mask], ordered.loc[valid_mask, "tstr_after"], color="#333333", linewidth=2, linestyle="--", label="After")
    ax2.fill_between(x[valid_mask], ordered.loc[valid_mask, "tstr_before"], ordered.loc[valid_mask, "tstr_after"], color="#FFB6C1", alpha=0.4, label="Utility Cost of Debiasing")
    ax2.axhline(refs["Real"]["TSTR_Accuracy"], color=COLORS["Real"], linestyle="--", label="Real Baseline")
    ax2.set_title("TSTR Accuracy: Before vs After Debiasing")
    ax2.set_ylabel("TSTR Accuracy (%)")

    skipped = ordered.loc[~valid_mask]
    for _, row in skipped.iterrows():
        idx = VARIANT_ORDER_STANDARD.index(row["variant"])
        ax1.scatter(idx, row["dp_before"], color="grey", marker="x", s=100)
        ax1.text(idx, row["dp_before"], " skipped", color="grey")
        ax2.scatter(idx, row["tstr_before"], color="grey", marker="x", s=100)
        ax2.text(idx, row["tstr_before"], " skipped", color="grey")

    for axis in (ax1, ax2):
        axis.set_xticks(x)
        axis.set_xticklabels([VARIANT_LABELS[variant] for variant in VARIANT_ORDER_STANDARD], rotation=20)
        axis.legend()

    fig.suptitle("Post-Hoc Debiasing: Can We Recover Fairness Lost to DP?")
    save_figure(fig, path)


def main(argv: list[str] | None = None) -> int:
    """Inputs: optional CLI argv. Outputs: integer exit code after figure generation. Lifecycle stage: Stage 6 — Visualisation. Reference: Direction 3 orchestration contract."""
    del argv
    paths = result_paths()
    dashboard_df = pd.read_csv(paths["dashboard"])
    refs = load_reference_rows()
    plot_figure4(dashboard_df, refs)
    plot_figure5(dashboard_df)
    plot_figure6(dashboard_df, refs)
    if paths["posthoc"].exists():
        posthoc_df = pd.read_csv(paths["posthoc"])
        plot_figure7(posthoc_df, refs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
