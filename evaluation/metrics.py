"""
metrics.py — TSTR accuracy, JS divergence, and final evaluation dashboard.

Run with:  python -m evaluation.metrics
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ── Paths & constants ────────────────────────────────────────────────

TRAIN_PATH = "data/adult_train.csv"
TEST_PATH = "data/adult_test.csv"
CTGAN_PATH = "results/ctgan_synthetic.csv"
TVAE_PATH = "results/tvae_synthetic.csv"
PRIVACY_PATH = "results/privacy_fairness_results.csv"
RESULTS_DIR = "results"

TARGET = "income"

CAT_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]

NUM_COLS = [
    "age", "fnlwgt", "education-num", "capital-gain",
    "capital-loss", "hours-per-week",
]


# ── Helpers ──────────────────────────────────────────────────────────

def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


# ── METRIC 1: TSTR Accuracy ─────────────────────────────────────────

def tstr_accuracy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> float:
    """Train on train_df, test on test_df. Returns accuracy (0–100)."""
    train_enc = _encode_categoricals(train_df)
    test_enc = _encode_categoricals(test_df)

    X_train = train_enc.drop(columns=[TARGET])
    y_train = train_enc[TARGET]
    X_test = test_enc.drop(columns=[TARGET])
    y_test = test_enc[TARGET]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test) * 100.0


# ── METRIC 2: JS Divergence ─────────────────────────────────────────

def _kl_divergence(p: np.ndarray, m: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / m)))


def js_divergence_column(
    real_vals: np.ndarray, synth_vals: np.ndarray, n_bins: int = 50
) -> float:
    """JS divergence for a single numerical column."""
    lo = min(real_vals.min(), synth_vals.min())
    hi = max(real_vals.max(), synth_vals.max())
    bins = np.linspace(lo, hi, n_bins + 1)

    eps = 1e-10
    p = np.histogram(real_vals, bins=bins)[0].astype(float)
    q = np.histogram(synth_vals, bins=bins)[0].astype(float)

    p = p / p.sum() + eps
    q = q / q.sum() + eps
    # re-normalise after adding eps
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def mean_js_divergence(
    real_df: pd.DataFrame, synth_df: pd.DataFrame
) -> float:
    """Average JS divergence across all numerical columns."""
    scores = []
    for col in NUM_COLS:
        js = js_divergence_column(
            real_df[col].values.astype(float),
            synth_df[col].values.astype(float),
        )
        scores.append(js)
    return float(np.mean(scores))


# ── Main ─────────────────────────────────────────────────────────────

def main():
    real_train = pd.read_csv(TRAIN_PATH)
    real_test = pd.read_csv(TEST_PATH)
    synth_ctgan = pd.read_csv(CTGAN_PATH)
    synth_tvae = pd.read_csv(TVAE_PATH)

    # Metric 1 — TSTR
    print("Computing TSTR accuracy (Real→Real) …")
    tstr_real = tstr_accuracy(real_train, real_test)
    print("Computing TSTR accuracy (CTGAN→Real) …")
    tstr_ctgan = tstr_accuracy(synth_ctgan, real_test)
    print("Computing TSTR accuracy (TVAE→Real) …")
    tstr_tvae = tstr_accuracy(synth_tvae, real_test)

    # Metric 2 — JS Divergence
    print("Computing JS divergence (CTGAN) …")
    js_ctgan = mean_js_divergence(real_train, synth_ctgan)
    print("Computing JS divergence (TVAE) …")
    js_tvae = mean_js_divergence(real_train, synth_tvae)

    # Metrics 3 & 4 — Load privacy/fairness
    pf = pd.read_csv(PRIVACY_PATH)
    ctgan_row = pf[pf["Model"] == "CTGAN"].iloc[0]
    tvae_row = pf[pf["Model"] == "TVAE"].iloc[0]
    real_row = pf[pf["Model"] == "Real"].iloc[0]

    mia_ctgan = ctgan_row["MIA_Advantage"]
    mia_tvae = tvae_row["MIA_Advantage"]
    dp_real = real_row["Demographic_Parity"]
    dp_ctgan = ctgan_row["Demographic_Parity"]
    dp_tvae = tvae_row["Demographic_Parity"]

    # Dashboard
    print("=" * 65)
    print("         FINAL EVALUATION DASHBOARD")
    print("=" * 65)
    print(f"{'Model':<10} {'JS Div':>10} {'TSTR Acc':>10} {'MIA Adv':>10} {'Demo.Parity':>12}")
    print("-" * 65)
    print(f"{'Real':<10} {'N/A':>10} {tstr_real:>9.1f}% {'N/A':>10} {dp_real:>11.4f}")
    print(f"{'CTGAN':<10} {js_ctgan:>10.4f} {tstr_ctgan:>9.1f}% {mia_ctgan:>10.4f} {dp_ctgan:>11.4f}")
    print(f"{'TVAE':<10} {js_tvae:>10.4f} {tstr_tvae:>9.1f}% {mia_tvae:>10.4f} {dp_tvae:>11.4f}")
    print("=" * 65)

    # Save
    results_df = pd.DataFrame([
        {"Model": "Real",  "JS_Divergence": None,     "TSTR_Accuracy": tstr_real,  "MIA_Advantage": None,      "Demographic_Parity": dp_real},
        {"Model": "CTGAN", "JS_Divergence": js_ctgan,  "TSTR_Accuracy": tstr_ctgan, "MIA_Advantage": mia_ctgan,  "Demographic_Parity": dp_ctgan},
        {"Model": "TVAE",  "JS_Divergence": js_tvae,   "TSTR_Accuracy": tstr_tvae,  "MIA_Advantage": mia_tvae,   "Demographic_Parity": dp_tvae},
    ])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "final_evaluation_table.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
