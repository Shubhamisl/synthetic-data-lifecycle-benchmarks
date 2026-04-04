"""
loader.py - Stage 1: Data acquisition, preprocessing, and summary.

Downloads the UCI Adult Income dataset, cleans it, encodes the target
as binary, splits into train/test, saves CSVs, and prints a full summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

import config


def load_and_preprocess() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch, clean, encode, split, save, and summarise the Adult dataset.

    Inputs: None.
    Outputs: Train and test dataframes with binary-encoded `income`, saved to disk.
    Lifecycle stage: Stage 1 - Data acquisition and preprocessing.
    Reference: Kohavi (1996) UCI Adult dataset usage for income prediction.
    """
    print("=" * 60)
    print("STAGE 1 - Data Acquisition & Preprocessing")
    print("=" * 60)

    print("\n[1] Fetching UCI Adult Income dataset (id=2) ...")
    adult = fetch_ucirepo(id=config.UCI_DATASET_ID)
    df = adult.data.features.copy()
    df[config.FAIRNESS_TARGET] = adult.data.targets.iloc[:, 0]
    print(f"    Raw shape: {df.shape}")

    print("[2] Dropping rows with missing values ...")
    df.replace("?", np.nan, inplace=True)
    n_before = len(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"    Dropped {n_before - len(df)} rows -> {df.shape}")

    print("[3] Encoding income as binary (0 = <=50K, 1 = >50K) ...")
    df[config.FAIRNESS_TARGET] = (
        df[config.FAIRNESS_TARGET]
        .str.strip()
        .str.rstrip(".")
        .map(lambda value: 1 if value == ">50K" else 0)
    )

    print("[4] Splitting into train/test (stratified by income) ...")
    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=df[config.FAIRNESS_TARGET],
    )
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = config.DATA_DIR / "adult_train.csv"
    test_path = config.DATA_DIR / "adult_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"    Saved -> {train_path}  ({len(train_df)} rows)")
    print(f"    Saved -> {test_path}  ({len(test_df)} rows)")

    _print_summary(train_df, test_df)
    return train_df, test_df


def identify_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Identify numerical and categorical columns in a dataframe.

    Inputs: Dataframe to inspect.
    Outputs: Tuple of (continuous_columns, discrete_columns).
    Lifecycle stage: Stage 1 - Data profiling.
    Reference: Standard pandas dtype-based schema inspection.
    """
    discrete = df.select_dtypes(include=["object", "category"]).columns.tolist()
    continuous = df.select_dtypes(include="number").columns.tolist()
    return continuous, discrete


def _print_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Print dataset summary for the processed Adult train/test split."""
    print("\n" + "-" * 60)
    print("DATA SUMMARY")
    print("-" * 60)

    print(f"\n  Train shape : {train_df.shape}")
    print(f"  Test  shape : {test_df.shape}")

    target = config.FAIRNESS_TARGET
    print("\n  Income distribution (train):")
    dist = train_df[target].value_counts().sort_index()
    for label, count in dist.items():
        tag = "<=50K" if label == 0 else ">50K"
        print(f"    {tag} ({label}): {count:>6}  ({count / len(train_df) * 100:.1f}%)")

    sens = config.FAIRNESS_SENSITIVE_FEATURE
    print(f"\n  Sensitive attribute '{sens}' distribution (train):")
    for value, count in train_df[sens].value_counts().items():
        print(f"    {value}: {count:>6}  ({count / len(train_df) * 100:.1f}%)")

    continuous, discrete = identify_columns(train_df)
    print(f"\n  Numerical columns  ({len(continuous)}): {continuous}")
    print(f"  Categorical columns ({len(discrete)}): {discrete}")
    print("-" * 60)


def main() -> None:
    """
    Execute Adult dataset download and preprocessing as a module entrypoint.

    Inputs: None.
    Outputs: Writes processed Adult train/test CSVs and prints a summary.
    Lifecycle stage: Stage 1 - Data acquisition and preprocessing.
    Reference: Kohavi (1996) UCI Adult dataset usage for income prediction.
    """
    load_and_preprocess()


if __name__ == "__main__":
    main()
