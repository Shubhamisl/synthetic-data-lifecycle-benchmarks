"""
train_models.py — Stage 2: Train CTGAN and TVAE, generate synthetic data, validate.
"""

import pickle

import pandas as pd
from ctgan import CTGAN
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer

import config

# ── Constants ────────────────────────────────────────────────────────

TRAIN_PATH = config.DATA_DIR / "adult_train.csv"
RESULTS_DIR = config.RESULTS_DIR
MODELS_DIR = config.PROJECT_ROOT / "models"

CATEGORICAL_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country", "income",
]

N_SYNTHETIC = 10_000


# ── CTGAN ────────────────────────────────────────────────────────────

def train_ctgan(train_df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("MODEL 1 — CTGAN")
    print("=" * 60)

    model = CTGAN(epochs=300, batch_size=500, verbose=True)
    model.fit(train_df, discrete_columns=CATEGORICAL_COLS)

    synth = model.sample(N_SYNTHETIC)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    synth.to_csv(RESULTS_DIR / "ctgan_synthetic.csv", index=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "ctgan_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("CTGAN training complete ✓")
    print(f"  Shape: {synth.shape}")
    print(synth.head())
    return synth


# ── TVAE ─────────────────────────────────────────────────────────────

def train_tvae(train_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("MODEL 2 — TVAE")
    print("=" * 60)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_df)
    metadata.update_column(column_name="income", sdtype="categorical")
    metadata.update_column(column_name="sex", sdtype="categorical")

    model = TVAESynthesizer(metadata, epochs=300)
    model.fit(train_df)

    synth = model.sample(N_SYNTHETIC)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    synth.to_csv(RESULTS_DIR / "tvae_synthetic.csv", index=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "tvae_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("TVAE training complete ✓")
    print(f"  Shape: {synth.shape}")
    print(synth.head())
    return synth


# ── Validation ───────────────────────────────────────────────────────

def validate(name: str, synth: pd.DataFrame) -> None:
    print(f"\n--- Validating {name} ---")
    ok = True

    # 1. Shape
    if synth.shape == (N_SYNTHETIC, 15):
        print(f"  ✓ Shape is {synth.shape}")
    else:
        print(f"  ✗ Shape is {synth.shape}, expected ({N_SYNTHETIC}, 15)")
        ok = False

    # 2. No NaNs
    nan_count = synth.isna().sum().sum()
    if nan_count == 0:
        print("  ✓ No NaN values")
    else:
        print(f"  ✗ Found {nan_count} NaN values")
        ok = False

    # 3. Income values
    income_vals = set(synth["income"].unique())
    if income_vals <= {0, 1}:
        print("  ✓ income contains only 0 and 1")
    else:
        print(f"  ✗ income contains unexpected values: {income_vals}")
        ok = False

    # 4. Sex values
    sex_vals = set(synth["sex"].unique())
    if sex_vals <= {"Male", "Female"}:
        print("  ✓ sex contains only Male and Female")
    else:
        print(f"  ✗ sex contains unexpected values: {sex_vals}")
        ok = False

    # 5. Income distribution
    print(f"\n  Income distribution ({name}):")
    for val, count in synth["income"].value_counts().sort_index().items():
        print(f"    {val}: {count}  ({count / len(synth) * 100:.1f}%)")

    # 6. Sex distribution
    print(f"\n  Sex distribution ({name}):")
    for val, count in synth["sex"].value_counts().items():
        print(f"    {val}: {count}  ({count / len(synth) * 100:.1f}%)")

    if ok:
        print(f"  ✓ {name} validation passed")
    else:
        print(f"  ✗ {name} validation had issues (see above)")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"Loaded training data: {train_df.shape}\n")

    synth_ctgan = train_ctgan(train_df)
    synth_tvae = train_tvae(train_df)

    validate("CTGAN", synth_ctgan)
    validate("TVAE", synth_tvae)

    print("\n" + "=" * 60)
    print("Both models trained and validated ✓")
    print("Synthetic data saved to results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
