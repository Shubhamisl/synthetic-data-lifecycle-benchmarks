"""Direction 3 evaluation for privacy-fairness-fidelity trade-offs."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

import config
from benchmarks.evaluate_benchmarks import (
    demographic_parity_difference as benchmark_demographic_parity_difference,
    mean_js_divergence as benchmark_mean_js_divergence,
    membership_inference_advantage as benchmark_membership_inference_advantage,
    tstr_accuracy as benchmark_tstr_accuracy,
    _numeric_feature_columns as numeric_feature_columns,
)
from dp_triangle.common import dataset_input_paths, dataset_result_dir, get_dataset_spec
from evaluation import metrics as base_metrics
from evaluation import privacy_fairness as base_privacy

torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)

VARIANT_DISPLAY = {
    "no_dp": "No DP",
    "eps_10": "epsilon=10.0",
    "eps_1": "epsilon=1.0",
    "eps_0_5": "epsilon=0.5",
    "eps_0_1": "epsilon=0.1",
}
VARIANT_ORDER = ["no_dp", "eps_10", "eps_1", "eps_0_5", "eps_0_1"]


def results_paths(dataset_name: str = "adult") -> dict[str, Path]:
    """
    Return canonical Direction 3 result paths for one dataset.

    Inputs: dataset name.
    Outputs: dashboard, subgroup, and dry-run marker paths.
    Lifecycle stage: Stage 4 - Evaluation.
    Reference: project orchestration design generalized to supporting datasets.
    """
    result_root = dataset_result_dir(dataset_name)
    return {
        "dashboard": result_root / "dp_triangle_dashboard.csv",
        "subgroup": result_root / "dp_subgroup_fairness.csv",
        "dry_run_marker": result_root / "dp_direction3_dry_run.marker",
    }


def synthetic_path(variant: str, dataset_name: str = "adult") -> Path:
    """
    Return the synthetic CSV path for a dataset variant.

    Inputs: variant key and dataset name.
    Outputs: synthetic CSV path.
    Lifecycle stage: Stage 4 - Evaluation.
    Reference: project orchestration design generalized to supporting datasets.
    """
    return dataset_result_dir(dataset_name) / f"dp_synthetic_{variant}.csv"


def derive_triangle_scores(
    tstr_accuracy: float,
    mia_advantage: float,
    demographic_parity: float,
    synthetic_positive_rate: float | None = None,
    real_positive_rate: float | None = None,
) -> dict[str, object]:
    """
    Normalize privacy, utility, fairness, and collapse-aware triangle scores.

    Inputs: TSTR percentage, MIA advantage, demographic parity, and optional positive-class rates.
    Outputs: normalized score dictionary with collapse flags.
    Lifecycle stage: Stage 4 - Evaluation.
    Reference: Direction 3 triangle formulation plus collapse-aware reporting guardrails.
    """
    privacy_score = 1.0 - mia_advantage
    utility_score = tstr_accuracy / 100.0
    fairness_score = math.nan if pd.isna(demographic_parity) else 1.0 - demographic_parity
    triangle_score = (
        math.nan
        if math.isnan(fairness_score)
        else (privacy_score + utility_score + fairness_score) / 3.0
    )
    if synthetic_positive_rate is None or real_positive_rate is None or real_positive_rate == 0:
        positive_class_retention = math.nan
    else:
        positive_class_retention = synthetic_positive_rate / real_positive_rate
    retention_penalty = (
        math.nan
        if pd.isna(positive_class_retention)
        else max(0.0, min(float(positive_class_retention), 1.0))
    )
    collapsed_minority_class = (
        synthetic_positive_rate is not None
        and math.isclose(float(synthetic_positive_rate), 0.0, abs_tol=1e-12)
    )
    collapse_reason = "positive_class_missing" if collapsed_minority_class else ""
    triangle_score_adjusted = (
        math.nan
        if math.isnan(triangle_score) or pd.isna(retention_penalty)
        else triangle_score * retention_penalty
    )
    return {
        "Privacy_Score": privacy_score,
        "Utility_Score": utility_score,
        "Fairness_Score": fairness_score,
        "Triangle_Score": triangle_score,
        "Synthetic_Positive_Rate": synthetic_positive_rate,
        "Positive_Class_Retention": positive_class_retention,
        "Collapsed_Minority_Class": collapsed_minority_class,
        "Collapse_Reason": collapse_reason,
        "Triangle_Score_Adjusted": triangle_score_adjusted,
    }


def subgroup_tstr_accuracy(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    """
    Compute TSTR accuracy on a filtered real test subset.

    Inputs: synthetic train dataframe and filtered real test dataframe.
    Outputs: subgroup TSTR accuracy percentage.
    Lifecycle stage: Stage 4 - Evaluation.
    Reference: project baseline RandomForest TSTR from evaluation.metrics.
    """
    train_enc = base_metrics._encode_categoricals(train_df)
    test_enc = base_metrics._encode_categoricals(test_df)

    x_train = train_enc.drop(columns=[config.DP_TARGET_COL])
    y_train = train_enc[config.DP_TARGET_COL]
    x_test = test_enc.drop(columns=[config.DP_TARGET_COL])
    y_test = test_enc[config.DP_TARGET_COL]

    clf = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_SEED)
    clf.fit(x_train, y_train)
    return clf.score(x_test, y_test) * 100.0


def generic_subgroup_tstr_accuracy(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> float:
    """
    Compute TSTR accuracy on a filtered real test subset for non-Adult datasets.

    Inputs: synthetic train dataframe, filtered real test dataframe, and target column.
    Outputs: subgroup TSTR accuracy percentage.
    Lifecycle stage: Stage 4 - Evaluation.
    Reference: benchmark RandomForest TSTR implementation for supporting datasets.
    """
    return benchmark_tstr_accuracy(train_df, test_df, target_col)


def compute_variant_metrics(
    variant: str,
    epsilon_value: float | None,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synth_df: pd.DataFrame,
) -> dict[str, object]:
    """
    Compute one Adult dashboard row of Direction 3 metrics.

    Inputs: variant metadata, real Adult train/test, and synthetic dataframe.
    Outputs: one dashboard row dictionary.
    Lifecycle stage: Stage 4 - Evaluation.
    Reference: evaluation.metrics, evaluation.privacy_fairness, and the Adult Direction 3 subgroup specification.
    """
    js_value = base_metrics.mean_js_divergence(real_train, synth_df)
    tstr_value = base_metrics.tstr_accuracy(synth_df, real_test)
    mia_value = base_privacy.membership_inference_attack(real_train, synth_df)
    dp_value = base_privacy.demographic_parity(synth_df)
    real_positive_rate = float((real_train[config.DP_TARGET_COL] == config.DP_MINORITY_CLASS).mean())
    synthetic_positive_rate = float((synth_df[config.DP_TARGET_COL] == config.DP_MINORITY_CLASS).mean())

    female_test = real_test[real_test[config.DP_SENSITIVE_COL] == "Female"].reset_index(drop=True)
    high_income_test = real_test[real_test[config.DP_TARGET_COL] == config.DP_MINORITY_CLASS].reset_index(drop=True)

    tstr_female = subgroup_tstr_accuracy(synth_df, female_test)
    tstr_high_income = subgroup_tstr_accuracy(synth_df, high_income_test)
    female_degradation = tstr_value - tstr_female
    high_income_degradation = tstr_value - tstr_high_income

    score_values = derive_triangle_scores(
        tstr_accuracy=tstr_value,
        mia_advantage=mia_value,
        demographic_parity=dp_value,
        synthetic_positive_rate=synthetic_positive_rate,
        real_positive_rate=real_positive_rate,
    )
    return {
        "variant": variant,
        "epsilon_label": VARIANT_DISPLAY[variant],
        "epsilon_value": epsilon_value,
        "JS": js_value,
        "TSTR": tstr_value,
        "MIA_Advantage": mia_value,
        "Demo_Parity": dp_value,
        **score_values,
        "TSTR_Female": tstr_female,
        "TSTR_High_Income": tstr_high_income,
        "Female_Degradation": female_degradation,
        "High_Income_Degradation": high_income_degradation,
    }


def compute_generic_variant_metrics(
    spec,
    variant: str,
    epsilon_value: float | None,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synth_df: pd.DataFrame,
) -> dict[str, object]:
    """
    Compute one supporting-dataset dashboard row of Direction 3 metrics.

    Inputs: dataset spec, variant metadata, real train/test, and synthetic dataframe.
    Outputs: one dashboard row dictionary using generic subgroup labels.
    Lifecycle stage: Stage 4 - Evaluation.
    Reference: benchmark evaluation helpers adapted to Direction 3 subgroup and collapse-aware reporting.
    """
    js_value = benchmark_mean_js_divergence(
        real_train,
        synth_df,
        numeric_feature_columns(real_train, spec.target_col),
    )
    tstr_value = benchmark_tstr_accuracy(synth_df, real_test, spec.target_col)
    mia_value = benchmark_membership_inference_advantage(real_train, synth_df)
    dp_value = (
        benchmark_demographic_parity_difference(
            synth_df,
            spec.sensitive_attr,
            spec.target_col,
            positive_value=spec.positive_class,
        )
        if spec.sensitive_attr
        else float("nan")
    )
    real_positive_rate = None
    synthetic_positive_rate = None
    if spec.supports_full_triangle:
        real_positive_rate = float((real_train[spec.target_col] == spec.positive_class).mean())
        synthetic_positive_rate = float((synth_df[spec.target_col] == spec.positive_class).mean())

    tstr_sensitive_subgroup = float("nan")
    sensitive_subgroup_degradation = float("nan")
    if spec.sensitive_attr and spec.sensitive_subgroup_value is not None:
        sensitive_test = real_test[real_test[spec.sensitive_attr] == spec.sensitive_subgroup_value].reset_index(drop=True)
        tstr_sensitive_subgroup = generic_subgroup_tstr_accuracy(synth_df, sensitive_test, spec.target_col)
        sensitive_subgroup_degradation = tstr_value - tstr_sensitive_subgroup

    positive_test = real_test[real_test[spec.target_col] == spec.positive_class].reset_index(drop=True)
    tstr_positive_class = generic_subgroup_tstr_accuracy(synth_df, positive_test, spec.target_col)
    positive_class_degradation = tstr_value - tstr_positive_class

    score_values = derive_triangle_scores(
        tstr_accuracy=tstr_value,
        mia_advantage=mia_value,
        demographic_parity=dp_value,
        synthetic_positive_rate=synthetic_positive_rate,
        real_positive_rate=real_positive_rate,
    )
    return {
        "variant": variant,
        "epsilon_label": VARIANT_DISPLAY[variant],
        "epsilon_value": epsilon_value,
        "JS": js_value,
        "TSTR": tstr_value,
        "MIA_Advantage": mia_value,
        "Demo_Parity": dp_value,
        **score_values,
        "Sensitive_Subgroup_Label": spec.sensitive_subgroup_value if spec.sensitive_subgroup_value is not None else "",
        "Positive_Class_Label": str(spec.positive_class),
        "TSTR_Sensitive_Subgroup": tstr_sensitive_subgroup,
        "TSTR_Positive_Class": tstr_positive_class,
        "Sensitive_Subgroup_Degradation": sensitive_subgroup_degradation,
        "Positive_Class_Degradation": positive_class_degradation,
    }


def compute_dashboard(dataset_name: str = "adult") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the Direction 3 dashboard for one dataset.

    Inputs: dataset name.
    Outputs: dashboard dataframe and subgroup dataframe.
    Lifecycle stage: Stage 4 - Evaluation.
    Reference: Direction 3 metric specification, currently implemented for Adult with dataset-aware path routing.
    """
    spec = get_dataset_spec(dataset_name)
    data_paths = dataset_input_paths(dataset_name)
    real_train = pd.read_csv(data_paths["train"])
    real_test = pd.read_csv(data_paths["test"])

    rows = []
    for variant in VARIANT_ORDER:
        synth_df = pd.read_csv(synthetic_path(variant, dataset_name=dataset_name))
        if dataset_name == "adult":
            row = compute_variant_metrics(
                variant=variant,
                epsilon_value=config.DP_EPSILON_VALUES[variant],
                real_train=real_train,
                real_test=real_test,
                synth_df=synth_df,
            )
            row.update(
                {
                    "Sensitive_Subgroup_Label": spec.sensitive_subgroup_value,
                    "Positive_Class_Label": str(spec.positive_class),
                    "TSTR_Sensitive_Subgroup": row["TSTR_Female"],
                    "TSTR_Positive_Class": row["TSTR_High_Income"],
                    "Sensitive_Subgroup_Degradation": row["Female_Degradation"],
                    "Positive_Class_Degradation": row["High_Income_Degradation"],
                }
            )
        else:
            row = compute_generic_variant_metrics(
                spec=spec,
                variant=variant,
                epsilon_value=config.DP_EPSILON_VALUES[variant],
                real_train=real_train,
                real_test=real_test,
                synth_df=synth_df,
            )
        rows.append(row)

    dashboard_df = pd.DataFrame(rows)
    subgroup_df = pd.DataFrame()
    if spec.supports_full_triangle:
        subgroup_df = dashboard_df[
            [
                "variant",
                "TSTR",
                "TSTR_Sensitive_Subgroup",
                "TSTR_Positive_Class",
                "Sensitive_Subgroup_Degradation",
                "Positive_Class_Degradation",
            ]
        ].rename(
            columns={
                "TSTR": "tstr_overall",
                "TSTR_Sensitive_Subgroup": "tstr_sensitive_subgroup",
                "TSTR_Positive_Class": "tstr_positive_class",
                "Sensitive_Subgroup_Degradation": "sensitive_subgroup_degradation",
                "Positive_Class_Degradation": "positive_class_degradation",
            }
        )
    return dashboard_df, subgroup_df


def main(argv: list[str] | None = None) -> int:
    """
    Run Direction 3 evaluation for one dataset.

    Inputs: optional CLI argv where the first argument is the dataset name.
    Outputs: integer exit code after caching or evaluation.
    Lifecycle stage: Stage 4 - Evaluation.
    Reference: Direction 3 orchestration contract generalized to supporting datasets.
    """
    dataset_name = argv[0] if argv else "adult"
    get_dataset_spec(dataset_name)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    path_map = results_paths(dataset_name)
    if path_map["dashboard"].exists() and not path_map["dry_run_marker"].exists():
        print(f"[CACHE HIT] Skipping evaluation - {path_map['dashboard'].name} already exists")
        return 0

    path_map["dashboard"].parent.mkdir(parents=True, exist_ok=True)
    dashboard_df, subgroup_df = compute_dashboard(dataset_name)
    dashboard_df.to_csv(path_map["dashboard"], index=False)
    if subgroup_df.empty:
        path_map["subgroup"].unlink(missing_ok=True)
    else:
        subgroup_df.to_csv(path_map["subgroup"], index=False)
    print(dashboard_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
