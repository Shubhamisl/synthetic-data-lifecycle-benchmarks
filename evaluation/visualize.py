"""
visualize.py — Generate and save evaluation plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config


def plot_column_distributions(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: list[str] | None = None,
    tag: str = "ctgan",
):
    """Side-by-side histograms / count-plots for selected columns."""
    columns = columns or real.columns.tolist()
    n = len(columns)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i, col in enumerate(columns):
        for j, (data, label) in enumerate([(real, "Real"), (synthetic, "Synthetic")]):
            ax = axes[i][j]
            if data[col].dtype in ("object", "category"):
                sns.countplot(x=col, data=data, ax=ax, order=real[col].value_counts().index[:10])
            else:
                ax.hist(data[col].dropna(), bins=30, edgecolor="black", alpha=0.7)
            ax.set_title(f"{label} — {col}")
    plt.tight_layout()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"distributions_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved distribution plot → {path}")


def plot_correlation_heatmaps(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    tag: str = "ctgan",
):
    """Real vs synthetic correlation heatmaps for numeric columns."""
    num_cols = real.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        print("  Not enough numeric columns for correlation heatmap.")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(real[num_cols].corr(), annot=True, fmt=".2f", ax=ax1, cmap="coolwarm", vmin=-1, vmax=1)
    ax1.set_title("Real data")
    sns.heatmap(synthetic[num_cols].corr(), annot=True, fmt=".2f", ax=ax2, cmap="coolwarm", vmin=-1, vmax=1)
    ax2.set_title("Synthetic data")
    plt.tight_layout()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"correlations_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved correlation plot → {path}")


def save_metrics_table(metrics: dict, filename: str = "metrics_summary.csv"):
    """Flatten nested metrics dict into a CSV."""
    rows = []
    for section, values in metrics.items():
        if isinstance(values, dict):
            for k, v in values.items():
                rows.append({"category": section, "metric": k, "value": v})
        else:
            rows.append({"category": section, "metric": section, "value": values})
    df = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / filename
    df.to_csv(path, index=False)
    print(f"  Saved metrics table → {path}")
    return df
