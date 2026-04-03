"""
privacy_fairness.py — Membership Inference Attack & Demographic Parity evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config

# ── Paths ────────────────────────────────────────────────────────────

TRAIN_PATH = config.DATA_DIR / "adult_train.csv"
TEST_PATH = config.DATA_DIR / "adult_test.csv"
CTGAN_PATH = config.RESULTS_DIR / "ctgan_synthetic.csv"
TVAE_PATH = config.RESULTS_DIR / "tvae_synthetic.csv"
RESULTS_DIR = config.RESULTS_DIR

TARGET = "income"
SENSITIVE = "sex"

CAT_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]


# ── Helpers ──────────────────────────────────────────────────────────

def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all categorical columns in-place copy."""
    df = df.copy()
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


# ── PART A: Membership Inference Attack ──────────────────────────────

def membership_inference_attack(
    real_df: pd.DataFrame, synth_df: pd.DataFrame
) -> float:
    """
    Returns MIA Advantage = attack_accuracy - 0.5.
    Near 0.0 = strong privacy, near 0.5 = weak privacy.
    """
    real = real_df.copy()
    real["_member"] = 1
    synth = synth_df.copy()
    synth["_member"] = 0

    combined = pd.concat([real, synth], ignore_index=True).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    y = combined["_member"]
    X = _encode_categoricals(combined.drop(columns=["_member"]))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)

    return accuracy - 0.5


# ── PART B: Demographic Parity ───────────────────────────────────────

def demographic_parity(df: pd.DataFrame) -> float:
    """
    |P(income=1 | Male) - P(income=1 | Female)|
    0 = perfect fairness.
    """
    males = df[df[SENSITIVE] == "Male"]
    females = df[df[SENSITIVE] == "Female"]
    p_male = males[TARGET].mean()
    p_female = females[TARGET].mean()
    return abs(p_male - p_female)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    real_train = pd.read_csv(TRAIN_PATH)
    synth_ctgan = pd.read_csv(CTGAN_PATH)
    synth_tvae = pd.read_csv(TVAE_PATH)

    # Part A — MIA
    print("Running Membership Inference Attack (CTGAN) …")
    mia_ctgan = membership_inference_attack(real_train, synth_ctgan)
    print("Running Membership Inference Attack (TVAE) …")
    mia_tvae = membership_inference_attack(real_train, synth_tvae)

    # Part B — Demographic Parity
    dp_real = demographic_parity(real_train)
    dp_ctgan = demographic_parity(synth_ctgan)
    dp_tvae = demographic_parity(synth_tvae)

    # Output table
    print("=" * 55)
    print("      PRIVACY & FAIRNESS RESULTS")
    print("=" * 55)
    print(f"{'Model':<12} {'MIA Advantage':>15} {'Demo. Parity':>15}")
    print("-" * 55)
    print(f"{'Real':<12} {'N/A':>15} {dp_real:>14.4f}")
    print(f"{'CTGAN':<12} {mia_ctgan:>14.4f} {dp_ctgan:>14.4f}")
    print(f"{'TVAE':<12}  {mia_tvae:>14.4f} {dp_tvae:>14.4f}")
    print("=" * 55)

    # Save to CSV
    results_df = pd.DataFrame([
        {"Model": "Real",  "MIA_Advantage": None,      "Demographic_Parity": dp_real},
        {"Model": "CTGAN", "MIA_Advantage": mia_ctgan,  "Demographic_Parity": dp_ctgan},
        {"Model": "TVAE",  "MIA_Advantage": mia_tvae,   "Demographic_Parity": dp_tvae},
    ])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "privacy_fairness_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
