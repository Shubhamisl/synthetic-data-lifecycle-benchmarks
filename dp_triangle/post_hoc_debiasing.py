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
from dp_triangle.common import dataset_input_paths, dataset_result_dir, get_dataset_spec

torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)

VARIANT_ORDER = ["no_dp", "eps_10", "eps_1", "eps_0_5", "eps_0_1"]
VARIANT_LABELS = {
    "no_dp": "No DP",
    "eps_10": "epsilon=10.0",
    "eps_1": "epsilon=1.0",
    "eps_0_5": "epsilon=0.5",
    "eps_0_1": "epsilon=0.1",
}


def output_path(dataset_name: str = "adult") -> Path:
    """
    Return the post-hoc debiasing output path for one dataset.

    Inputs: dataset name.
    Outputs: post-hoc debiasing CSV path.
    Lifecycle stage: Stage 5 - Post-hoc Debiasing.
    Reference: Direction 3 output specification generalized to supporting datasets.
    """
    return dataset_result_dir(dataset_name) / "dp_post_hoc_debiasing.csv"


def dry_run_marker_path(dataset_name: str = "adult") -> Path:
    """
    Return the dry-run marker path for one dataset.

    Inputs: dataset name.
    Outputs: dry-run marker path.
    Lifecycle stage: Stage 5 - Post-hoc Debiasing.
    Reference: Direction 3 dry-run cache invalidation design generalized to supporting datasets.
    """
    return dataset_result_dir(dataset_name) / "dp_direction3_dry_run.marker"


def synthetic_path(variant: str, dataset_name: str = "adult") -> Path:
    """
    Return the synthetic CSV path for one dataset variant.

    Inputs: variant key and dataset name.
    Outputs: synthetic CSV path.
    Lifecycle stage: Stage 5 - Post-hoc Debiasing.
    Reference: Direction 3 output specification generalized to supporting datasets.
    """
    return dataset_result_dir(dataset_name) / f"dp_synthetic_{variant}.csv"


def encode_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ordinal-encode train and test features consistently.

    Inputs: synthetic train dataframe, real test dataframe, and target column name.
    Outputs: encoded feature dataframes.
    Lifecycle stage: Stage 5 - Post-hoc Debiasing.
    Reference: standard supervised tabular preprocessing for parity mitigation.
    """
    x_train = train_df.drop(columns=[target_col]).copy()
    x_test = test_df.drop(columns=[target_col]).copy()
    feature_cats = x_train.select_dtypes(include=["object", "category"]).columns.tolist()

    if feature_cats:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        x_train[feature_cats] = encoder.fit_transform(x_train[feature_cats].astype(str))
        x_test[feature_cats] = encoder.transform(x_test[feature_cats].astype(str))

    return x_train, x_test


def demographic_parity_from_predictions(predictions: np.ndarray, sensitive_series: pd.Series, group_a: str, group_b: str) -> float:
    """
    Compute demographic parity gap from predicted labels.

    Inputs: predictions, sensitive-feature series, and the two comparison group labels.
    Outputs: demographic parity gap.
    Lifecycle stage: Stage 5 - Post-hoc Debiasing.
    Reference: demographic parity difference definition from the project baseline.
    """
    sensitive = sensitive_series.reset_index(drop=True)
    pred_series = pd.Series(predictions).reset_index(drop=True)
    group_a_rate = pred_series[sensitive == group_a].mean()
    group_b_rate = pred_series[sensitive == group_b].mean()
    return abs(float(group_a_rate) - float(group_b_rate))


def main(argv: list[str] | None = None) -> int:
    """
    Run post-hoc debiasing for one dataset.

    Inputs: optional CLI argv where the first argument is the dataset name.
    Outputs: integer exit code after post-hoc debiasing evaluation.
    Lifecycle stage: Stage 5 - Post-hoc Debiasing.
    Reference: Direction 3 post-processing experiment generalized to supporting datasets.
    """
    dataset_name = argv[0] if argv else "adult"
    spec = get_dataset_spec(dataset_name)
    if not spec.supports_full_triangle or not spec.sensitive_attr or not spec.sensitive_groups:
        output_path(dataset_name).unlink(missing_ok=True)
        print(f"[SKIP] Post-hoc debiasing not applicable for {dataset_name}")
        return 0

    if output_path(dataset_name).exists() and not dry_run_marker_path(dataset_name).exists():
        print(f"[CACHE HIT] Skipping post-hoc debiasing - {output_path(dataset_name).name} already exists")
        return 0

    dashboard_df = pd.read_csv(dataset_result_dir(dataset_name) / "dp_triangle_dashboard.csv").set_index("variant")
    real_test = pd.read_csv(dataset_input_paths(dataset_name)["test"])

    rows = []
    for variant in VARIANT_ORDER:
        synth_df = pd.read_csv(synthetic_path(variant, dataset_name=dataset_name))
        x_synth, x_test = encode_train_test(synth_df, real_test, target_col=spec.target_col)
        y_synth = synth_df[spec.target_col]
        y_test = real_test[spec.target_col]
        sensitive_synth = synth_df[spec.sensitive_attr]
        sensitive_test = real_test[spec.sensitive_attr]

        dp_before = float(dashboard_df.loc[variant, "Demo_Parity"])
        tstr_before = float(dashboard_df.loc[variant, "TSTR"])

        base_clf = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_SEED)
        mitigator = ExponentiatedGradient(base_clf, DemographicParity())

        try:
            mitigator.fit(x_synth, y_synth, sensitive_features=sensitive_synth)
            predictions_after = mitigator.predict(x_test)
            dp_after = demographic_parity_from_predictions(
                np.asarray(predictions_after),
                sensitive_test,
                spec.sensitive_groups[0],
                spec.sensitive_groups[1],
            )
            tstr_after = float(accuracy_score(y_test, predictions_after) * 100.0)
            recovery = math.nan if dp_before == 0 else (dp_before - dp_after) / dp_before * 100.0
            utility_cost = tstr_before - tstr_after
            status = "success"
        except ValueError as exc:
            print(f"Post-hoc debiasing SKIPPED for {variant} - {exc}")
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
    out_df.to_csv(output_path(dataset_name), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
