from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .common import BENCHMARK_ROOT, DATASET_REGISTRY, get_dataset_paths
from .evaluate_benchmarks import demographic_parity_difference

RESULTS_DIR = BENCHMARK_ROOT / "results"
PLOTS_DIR = BENCHMARK_ROOT / "plots"
SUMMARY_PATH = RESULTS_DIR / "cross_domain_summary.csv"
MEAN_RANK_PATH = RESULTS_DIR / "mean_rank_table.csv"
DATASET_ORDER = ["adult", "bank", "covertype", "diabetes"]
MODEL_ORDER = ["CTGAN", "TVAE"]
REAL_COLOR = "#9E9E9E"
CTGAN_COLOR = "#FF9800"
TVAE_COLOR = "#4CAF50"
MODEL_COLORS = {"CTGAN": CTGAN_COLOR, "TVAE": TVAE_COLOR}
DATASET_MARKERS = {"adult": "*", "bank": "o", "covertype": "^", "diabetes": "s"}


def required_plot_paths() -> list[Path]:
    return [
        PLOTS_DIR / "plot1_tstr_heatmap.png",
        PLOTS_DIR / "plot2_cross_domain_dashboard.png",
        PLOTS_DIR / "plot3_mean_rank.png",
        PLOTS_DIR / "plot4_privacy_utility_all_domains.png",
    ]


def _configure_stdout_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def validate_plot_inputs(summary_df: pd.DataFrame, rank_df: pd.DataFrame) -> None:
    summary_required = {
        "Dataset",
        "Model",
        "JS_Divergence",
        "TSTR_Accuracy",
        "MIA_Advantage",
        "Demographic_Parity",
        "TSTR_Real_Baseline",
    }
    rank_required = {
        "Model",
        "Mean_TSTR_Rank",
        "Mean_JS_Rank",
        "Mean_MIA_Rank",
        "Mean_DP_Rank",
        "Overall_Mean_Rank",
    }

    if not summary_required.issubset(summary_df.columns):
        missing = sorted(summary_required - set(summary_df.columns))
        raise ValueError(f"missing columns in summary data: {missing}")
    if not rank_required.issubset(rank_df.columns):
        missing = sorted(rank_required - set(rank_df.columns))
        raise ValueError(f"missing columns in rank data: {missing}")

    expected_pairs = {(dataset, model) for dataset in DATASET_ORDER for model in MODEL_ORDER}
    observed_pairs = set(summary_df[["Dataset", "Model"]].itertuples(index=False, name=None))
    if observed_pairs != expected_pairs:
        raise ValueError("summary data is missing expected dataset/model combinations")


def _real_dp_baselines() -> dict[str, float]:
    baselines: dict[str, float] = {}
    for dataset_name in DATASET_ORDER:
        spec = DATASET_REGISTRY[dataset_name]
        if not spec.sensitive_attr:
            baselines[dataset_name] = float("nan")
            continue
        train_df = pd.read_csv(get_dataset_paths(dataset_name)["train"])
        baselines[dataset_name] = demographic_parity_difference(train_df, spec.sensitive_attr, spec.target_col)
    return baselines


def _summary_lookup(summary_df: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for row in summary_df.to_dict(orient="records"):
        lookup[(row["Dataset"], row["Model"])] = row
    return lookup


def build_tstr_heatmap(summary_df: pd.DataFrame, output_path: Path) -> None:
    lookup = _summary_lookup(summary_df)
    rows = []
    annotations = []
    for dataset in DATASET_ORDER:
        baseline = lookup[(dataset, "CTGAN")]["TSTR_Real_Baseline"]
        ctgan_value = lookup[(dataset, "CTGAN")]["TSTR_Accuracy"]
        tvae_value = lookup[(dataset, "TVAE")]["TSTR_Accuracy"]
        rows.append([baseline, ctgan_value, tvae_value])
        annotations.append(
            [
                f"{baseline:.2f}\n(+0.00)",
                f"{ctgan_value:.2f}\n({ctgan_value - baseline:+.2f})",
                f"{tvae_value:.2f}\n({tvae_value - baseline:+.2f})",
            ]
        )

    heatmap_df = pd.DataFrame(rows, index=DATASET_ORDER, columns=["Real Baseline", "CTGAN", "TVAE"])
    annot_df = pd.DataFrame(annotations, index=DATASET_ORDER, columns=heatmap_df.columns)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    sns.heatmap(
        heatmap_df,
        annot=annot_df,
        fmt="",
        cmap="RdYlGn",
        linewidths=0.5,
        cbar_kws={"label": "Accuracy (%)"},
        ax=ax,
    )
    ax.set_title("TSTR Accuracy Across Domains (%)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_metric_dashboard(summary_df: pd.DataFrame, output_path: Path) -> None:
    lookup = _summary_lookup(summary_df)
    real_dp = _real_dp_baselines()
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), dpi=300)
    axes = axes.flatten()
    datasets = DATASET_ORDER
    positions = list(range(len(datasets)))
    width = 0.22

    metric_specs = [
        ("JS_Divergence", "JS Div", ["CTGAN", "TVAE"]),
        ("TSTR_Accuracy", "TSTR Acc", ["Real", "CTGAN", "TVAE"]),
        ("MIA_Advantage", "MIA Adv", ["CTGAN", "TVAE"]),
        ("Demographic_Parity", "Demo Parity", ["Real", "CTGAN", "TVAE"]),
    ]

    for ax, (metric_key, title, series_order) in zip(axes, metric_specs):
        for series_index, series_name in enumerate(series_order):
            offsets = [pos + (series_index - (len(series_order) - 1) / 2) * width for pos in positions]
            values = []
            for dataset in datasets:
                if series_name == "Real":
                    if metric_key == "TSTR_Accuracy":
                        values.append(float(lookup[(dataset, "CTGAN")]["TSTR_Real_Baseline"]))
                    else:
                        values.append(real_dp[dataset])
                else:
                    values.append(float(lookup[(dataset, series_name)][metric_key]))

            color = REAL_COLOR if series_name == "Real" else MODEL_COLORS[series_name]
            bars = ax.bar(offsets, values, width=width, color=color, label=series_name)
            for bar, value in zip(bars, values):
                if pd.isna(value):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_xticks(positions)
        ax.set_xticklabels(datasets)
        ax.set_title(title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Cross-Domain Evaluation Dashboard")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_mean_rank_chart(rank_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    metric_columns = [
        "Mean_TSTR_Rank",
        "Mean_JS_Rank",
        "Mean_MIA_Rank",
        "Mean_DP_Rank",
    ]
    metric_labels = ["TSTR", "JS", "MIA", "DP"]
    offsets = [-0.27, -0.09, 0.09, 0.27]
    y_positions = list(range(len(rank_df)))

    for metric_label, metric_column, offset in zip(metric_labels, metric_columns, offsets):
        colors = []
        values = []
        y_vals = []
        for base_y, row in zip(y_positions, rank_df.to_dict(orient="records")):
            y_vals.append(base_y + offset)
            values.append(row[metric_column])
            base_color = MODEL_COLORS[row["Model"]]
            colors.append(base_color)
        bars = ax.barh(y_vals, values, height=0.16, color=colors, alpha=0.8, label=metric_label)
        for bar, value in zip(bars, values):
            ax.text(value, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", ha="left", fontsize=8)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(rank_df["Model"].tolist())
    ax.set_title("Mean Rank Across All Datasets (Lower is Better)")
    ax.set_xlabel("Rank")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_privacy_utility_scatter(summary_df: pd.DataFrame, output_path: Path) -> None:
    lookup = _summary_lookup(summary_df)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    ax.axvspan(83, ax.get_xlim()[1] if ax.get_xlim()[1] > 83 else 100, ymin=0, ymax=1, color="#dff5df", alpha=0.3)
    ax.axhspan(0, 0.15, xmin=0, xmax=1, color="#dff5df", alpha=0.3)

    baseline_positions = {}
    for dataset in DATASET_ORDER:
        baseline = float(lookup[(dataset, "CTGAN")]["TSTR_Real_Baseline"])
        baseline_positions[dataset] = baseline
        ax.axvline(baseline, color="lightgrey", linestyle="--", linewidth=1)

    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            row = lookup[(dataset, model)]
            x_value = float(row["TSTR_Accuracy"])
            y_value = float(row["MIA_Advantage"])
            ax.scatter(
                x_value,
                y_value,
                color=MODEL_COLORS[model],
                marker=DATASET_MARKERS[dataset],
                s=120,
            )
            ax.text(x_value, y_value, f"{dataset}\n{model}", fontsize=8, ha="left", va="bottom")

    ax.set_title("Privacy-Utility Trade-off Across All Domains")
    ax.set_xlabel("TSTR Accuracy (%) -> Higher is Better")
    ax.set_ylabel("MIA Advantage -> Lower is Better")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _configure_stdout_utf8()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    summary_df = pd.read_csv(SUMMARY_PATH)
    rank_df = pd.read_csv(MEAN_RANK_PATH)
    validate_plot_inputs(summary_df, rank_df)

    plot_paths = required_plot_paths()
    build_tstr_heatmap(summary_df, plot_paths[0])
    print("Plot 1 saved ✓")
    build_metric_dashboard(summary_df, plot_paths[1])
    print("Plot 2 saved ✓")
    build_mean_rank_chart(rank_df, plot_paths[2])
    print("Plot 3 saved ✓")
    build_privacy_utility_scatter(summary_df, plot_paths[3])
    print("Plot 4 saved ✓")


if __name__ == "__main__":
    main()
