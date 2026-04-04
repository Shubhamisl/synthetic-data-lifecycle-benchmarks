"""Post-hoc debiasing experiments for Direction 3."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder

import config

torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)

VARIANT_ORDER = ["no_dp", "eps_10", "eps_1", "eps_0_5", "eps_0_1"]
VARIANT_LABELS = {
    "no_dp": "No DP",
    "eps_10": "ε=10.0",
    "eps_1": "ε=1.0",
    "eps_0_5": "ε=0.5",
    "eps_0_1": "ε=0.1",
}


def output_path() -> Path:
    """Inputs: none. Outputs: post-hoc debiasing CSV path. Lifecycle stage: Stage 5 — Post-hoc Debiasing. Reference: Direction 3 output specification."""
    return config.RESULTS_DIR / "dp_post_hoc_debiasing.csv"


def dry_run_marker_path() -> Path:
    """Inputs: none. Outputs: dry-run marker path. Lifecycle stage: Stage 5 — Post-hoc Debiasing. Reference: Direction 3 dry-run cache invalidation design."""
    return config.RESULTS_DIR / "dp_direction3_dry_run.marker"


def synthetic_path(variant: str) -> Path:
    """Inputs: variant key. Outputs: synthetic CSV path. Lifecycle stage: Stage 5 — Post-hoc Debiasing. Reference: Direction 3 output specification."""
    return config.RESULTS_DIR / f"dp_synthetic_{variant}.csv"


def encode_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inputs: synthetic train dataframe and real test dataframe. Outputs: consistently ordinal-encoded feature dataframes. Lifecycle stage: Stage 5 — Post-hoc Debiasing. Reference: standard supervised tabular preprocessing for parity mitigation."""
    categorical_cols = train_df.select_dtypes(include=["object", "category"]).columns.tolist()
    x_train = train_df.drop(columns=[config.DP_TARGET_COL]).copy()
    x_test = test_df.drop(columns=[config.DP_TARGET_COL]).copy()

    if categorical_cols:
        feature_cats = [column for column in categorical_cols if column != config.DP_TARGET_COL]
        if feature_cats:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            x_train[feature_cats] = encoder.fit_transform(x_train[feature_cats].astype(str))
            x_test[feature_cats] = encoder.transform(x_test[feature_cats].astype(str))

    return x_train, x_test


def demographic_parity_from_predictions(predictions: np.ndarray, sensitive_series: pd.Series) -> float:
    """Inputs: binary predictions and sensitive-feature series. Outputs: demographic parity gap. Lifecycle stage: Stage 5 — Post-hoc Debiasing. Reference: demographic parity difference definition from the project baseline."""
    sensitive = sensitive_series.reset_index(drop=True)
    pred_series = pd.Series(predictions).reset_index(drop=True)
    male_rate = pred_series[sensitive == "Male"].mean()
    female_rate = pred_series[sensitive == "Female"].mean()
    return abs(float(male_rate) - float(female_rate))


def main(argv: list[str] | None = None) -> int:
    """Inputs: optional CLI argv. Outputs: integer exit code after post-hoc debiasing evaluation. Lifecycle stage: Stage 5 — Post-hoc Debiasing. Reference: Direction 3 post-processing experiment specification."""
    del argv
    if output_path().exists() and not dry_run_marker_path().exists():
        print(f"[CACHE HIT] Skipping post-hoc debiasing — {output_path().name} already exists")
        return 0

    dashboard_df = pd.read_csv(config.RESULTS_DIR / "dp_triangle_dashboard.csv").set_index("variant")
    real_test = pd.read_csv(config.DATA_DIR / "adult_test.csv")

    rows = []
    for variant in VARIANT_ORDER:
        synth_df = pd.read_csv(synthetic_path(variant))
        x_synth, x_test = encode_train_test(synth_df, real_test)
        y_synth = synth_df[config.DP_TARGET_COL]
        y_test = real_test[config.DP_TARGET_COL]
        sensitive_synth = synth_df[config.DP_SENSITIVE_COL]
        sensitive_test = real_test[config.DP_SENSITIVE_COL]

        dp_before = float(dashboard_df.loc[variant, "Demo_Parity"])
        tstr_before = float(dashboard_df.loc[variant, "TSTR"])

        base_clf = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_SEED)
        mitigator = ExponentiatedGradient(base_clf, DemographicParity())

        try:
            mitigator.fit(x_synth, y_synth, sensitive_features=sensitive_synth)
            predictions_after = mitigator.predict(x_test)
            dp_after = demographic_parity_from_predictions(np.asarray(predictions_after), sensitive_test)
            tstr_after = float(accuracy_score(y_test, predictions_after) * 100.0)
            recovery = math.nan if dp_before == 0 else (dp_before - dp_after) / dp_before * 100.0
            utility_cost = tstr_before - tstr_after
            status = "success"
        except ValueError as exc:
            print(f"Post-hoc debiasing SKIPPED for {variant} — {exc}")
            dp_after = math.nan
            tstr_after = math.nan
            recovery = math.nan
            utility_cost = math.nan
            status = f"skipped: {exc}"

        rows.append(
            {
                "variant": variant,
                "epsilon_label": VARIANT_LABELS[variant],
                "dp_before": dp_before,
                "dp_after": dp_after,
                "tstr_before": tstr_before,
                "tstr_after": tstr_after,
                "fairness_recovery_rate_%": recovery,
                "utility_cost": utility_cost,
                "debiasing_status": status,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path(), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
