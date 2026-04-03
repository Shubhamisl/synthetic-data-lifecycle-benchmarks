"""
visualize.py — Generate the 3 final evaluation plots for the dashboard.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

# Hardcoded evaluation results (from Stage 3 output)
results = {
    'Model':            ['Real',   'CTGAN',  'TVAE'],
    'JS_Divergence':    [None,      0.0527,   0.0345],
    'TSTR_Accuracy':    [85.42,     81.43,    82.19],
    'MIA_Advantage':    [0.0,       0.4896,   0.3756],  # N/A -> 0.0 for plot
    'Demographic_Parity':[0.2003,   0.2197,   0.1810]
}

COLORS = {
    'Real':  '#2196F3',  # blue
    'CTGAN': '#FF9800',  # orange
    'TVAE':  '#4CAF50'   # green
}

RESULTS_DIR = config.RESULTS_DIR
plt.style.use('seaborn-v0_8-whitegrid')


# ── Plot 1: Grouped Bar Chart ───────────────────────────────────────

def plot_dashboard():
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), dpi=300)
    
    models = results['Model']
    x = np.arange(len(models))
    
    # Extract data
    tstr_data = results['TSTR_Accuracy']
    mia_data = results['MIA_Advantage']
    dp_data = results['Demographic_Parity']
    
    # Subplot 1: TSTR Accuracy
    ax1 = axes[0]
    bars1 = ax1.bar(x, tstr_data, color=[COLORS[m] for m in models], edgecolor='white', linewidth=0.5)
    ax1.set_ylim(78, 88)
    ax1.axhline(y=85.42, color='red', linestyle='--', alpha=0.7, label='Real Baseline')
    ax1.bar_label(bars1, fmt='%.2f', padding=3, fontsize=10)
    ax1.set_title("TSTR Accuracy (%)\n↑ Higher is Better", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(loc='lower left', fontsize=10)
    
    # Subplot 2: MIA Advantage
    ax2 = axes[1]
    bars2 = ax2.bar(x, mia_data, color=[COLORS[m] for m in models], edgecolor='white', linewidth=0.5)
    ax2.set_ylim(0.0, 0.6)
    ax2.axhline(y=0.0, color='green', linestyle='--', alpha=0.7, label='Ideal (0.0)')
    ax2.bar_label(bars2, fmt='%.3f', padding=3, fontsize=10)
    ax2.set_title("MIA Advantage\n↓ Lower is Better", fontsize=12)
    ax2.set_ylabel("MIA Advantage Score", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend(loc='upper right', fontsize=10)
    
    # Subplot 3: Demographic Parity
    ax3 = axes[2]
    bars3 = ax3.bar(x, dp_data, color=[COLORS[m] for m in models], edgecolor='white', linewidth=0.5)
    ax3.set_ylim(0.0, 0.30)
    ax3.axhline(y=0.0, color='green', linestyle='--', alpha=0.7, label='Perfect Fairness')
    ax3.bar_label(bars3, fmt='%.4f', padding=3, fontsize=10)
    ax3.set_title("Demographic Parity\n↓ Lower is Better", fontsize=12)
    ax3.set_ylabel("Parity Difference", fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.legend(loc='upper right', fontsize=10)
    
    # Figure level formatting
    fig.suptitle("Multi-Dimensional Synthetic Data Evaluation Dashboard", fontsize=16, y=0.98)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS[m], edgecolor='white', label=m) for m in models]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=3, fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    path = RESULTS_DIR / "plot1_evaluation_dashboard.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Plot 1 saved: {path} ✓")


# ── Plot 2: Distribution Comparison ──────────────────────────────────

def compute_js(p_vals, q_vals, bins=50):
    lo = min(p_vals.min(), q_vals.min())
    hi = max(p_vals.max(), q_vals.max())
    b = np.linspace(lo, hi, bins + 1)
    
    p = np.histogram(p_vals, bins=b)[0].astype(float)
    q = np.histogram(q_vals, bins=b)[0].astype(float)
    
    eps = 1e-10
    p = p / p.sum() + eps
    q = q / q.sum() + eps
    p = p / p.sum()
    q = q / q.sum()
    
    m = 0.5 * (p + q)
    kl_p = np.sum(p * np.log(p / m))
    kl_q = np.sum(q * np.log(q / m))
    return 0.5 * kl_p + 0.5 * kl_q

def plot_distributions():
    # Load actual datasets
    real = pd.read_csv(config.DATA_DIR / "adult_train.csv")
    ctgan = pd.read_csv(config.RESULTS_DIR / "ctgan_synthetic.csv")
    tvae = pd.read_csv(config.RESULTS_DIR / "tvae_synthetic.csv")
    
    cols = ["age", "hours-per-week", "capital-gain", "education-num"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes = axes.flatten()
    
    for i, col in enumerate(cols):
        ax = axes[i]
        
        r_vals = real[col].values
        c_vals = ctgan[col].values
        t_vals = tvae[col].values
        
        # Calculate JS runtime
        js_ctgan = compute_js(r_vals, c_vals, bins=30)
        js_tvae = compute_js(r_vals, t_vals, bins=30)
        
        ax.hist(r_vals, bins=30, alpha=0.5, color=COLORS['Real'], label='Real', density=True)
        ax.hist(c_vals, bins=30, alpha=0.5, color=COLORS['CTGAN'], label='CTGAN', density=True)
        ax.hist(t_vals, bins=30, alpha=0.5, color=COLORS['TVAE'], label='TVAE', density=True)
        
        ax.set_title(col, fontsize=12)
        
        # Annotate JS box
        text = f"JS: CTGAN={js_ctgan:.2f} | TVAE={js_tvae:.2f}"
        ax.text(0.95, 0.95, text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    fig.suptitle("Real vs Synthetic Data Distribution Comparison", fontsize=16, y=0.98)
    # One shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=3, fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    path = RESULTS_DIR / "plot2_distribution_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Plot 2 saved: {path} ✓")


# ── Plot 3: Privacy-Utility Tradeoff ─────────────────────────────────

def plot_tradeoff():
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    
    # Points
    ax.scatter([85.42], [0.0],    color=COLORS['Real'],  marker='*', s=300, label='Real')
    ax.scatter([81.43], [0.4896], color=COLORS['CTGAN'], marker='o', s=200, label='CTGAN')
    ax.scatter([82.19], [0.3756], color=COLORS['TVAE'],  marker='s', s=200, label='TVAE')
    
    # Text labels
    ax.text(85.42 + 0.1, 0.0 + 0.01,    " Real", fontsize=11, color=COLORS['Real'], weight='bold')
    ax.text(81.43 + 0.1, 0.4896 + 0.01, " CTGAN", fontsize=11, color=COLORS['CTGAN'], weight='bold')
    ax.text(82.19 + 0.1, 0.3756 + 0.01, " TVAE", fontsize=11, color=COLORS['TVAE'], weight='bold')
    
    # Ideal Zone
    import matplotlib.patches as patches
    rect = patches.Rectangle((83, 0), 4, 0.15, linewidth=1, edgecolor='none', 
                             facecolor='#4CAF50', alpha=0.2)
    ax.add_patch(rect)
    ax.text(85.0, 0.075, "Ideal Zone", fontsize=12, color='#2E7D32', 
            ha='center', va='center', weight='bold')
    
    # Arrow
    ax.annotate('', xy=(83.5, 0.16), xytext=(81.6, 0.45),
                arrowprops=dict(facecolor='gray', shrink=0.05, linestyle='--', 
                                width=1, headwidth=8, alpha=0.6))
    
    ax.set_xlim(79, 87)
    ax.set_ylim(-0.05, 0.6)
    
    ax.set_xlabel("TSTR Accuracy (%) → Higher is Better", fontsize=12)
    ax.set_ylabel("MIA Advantage → Lower is Better", fontsize=12)
    ax.set_title("Privacy-Utility Trade-off Analysis", fontsize=15, pad=15)
    
    plt.tight_layout()
    path = RESULTS_DIR / "plot3_privacy_utility_tradeoff.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Plot 3 saved: {path} ✓")


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_dashboard()
    plot_distributions()
    plot_tradeoff()
    print("All visualizations complete ✓")
