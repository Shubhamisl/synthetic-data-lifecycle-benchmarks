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
from evaluation import metrics as base_metrics
from evaluation import privacy_fairness as base_privacy

torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)

VARIANT_DISPLAY = {
    "no_dp": "No DP",
    "eps_10": "ε=10.0",
    "eps_1": "ε=1.0",
    "eps_0_5": "ε=0.5",
    "eps_0_1": "ε=0.1",
}
VARIANT_ORDER = ["no_dp", "eps_10", "eps_1", "eps_0_5", "eps_0_1"]


def results_paths() -> dict[str, Path]:
    """Inputs: none. Outputs: canonical Direction 3 result paths. Lifecycle stage: Stage 4 — Evaluation. Reference: project orchestration design."""
    return {
        "dashboard": config.RESULTS_DIR / "dp_triangle_dashboard.csv",
        "subgroup": config.RESULTS_DIR / "dp_subgroup_fairness.csv",
        "dry_run_marker": config.RESULTS_DIR / "dp_direction3_dry_run.marker",
    }


def synthetic_path(variant: str) -> Path:
    """Inputs: variant key. Outputs: synthetic CSV path. Lifecycle stage: Stage 4 — Evaluation. Reference: project orchestration design."""
    return config.RESULTS_DIR / f"dp_synthetic_{variant}.csv"


def derive_triangle_scores(
    tstr_accuracy: float,
    mia_advantage: float,
    demographic_parity: float,
) -> dict[str, float]:
    """Inputs: TSTR percentage, MIA advantage, and demographic parity. Outputs: normalized privacy, utility, fairness, and triangle scores. Lifecycle stage: Stage 4 — Evaluation. Reference: synthetic data multi-objective ranking formulation adapted from the Direction 3 spec."""
    privacy_score = 1.0 - mia_advantage
    utility_score = tstr_accuracy / 100.0
    fairness_score = math.nan if pd.isna(demographic_parity) else 1.0 - demographic_parity
    triangle_score = (
        math.nan
        if math.isnan(fairness_score)
        else (privacy_score + utility_score + fairness_score) / 3.0
    )
    return {
        "Privacy_Score": privacy_score,
        "Utility_Score": utility_score,
        "Fairness_Score": fairness_score,
        "Triangle_Score": triangle_score,
    }


def subgroup_tstr_accuracy(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    """Inputs: synthetic train dataframe and filtered real test dataframe. Outputs: subgroup TSTR accuracy percentage. Lifecycle stage: Stage 4 — Evaluation. Reference: project baseline RandomForest TSTR from `evaluation.metrics`."""
    train_enc = base_metrics._encode_categoricals(train_df)
    test_enc = base_metrics._encode_categoricals(test_df)

    x_train = train_enc.drop(columns=[config.DP_TARGET_COL])
    y_train = train_enc[config.DP_TARGET_COL]
    x_test = test_enc.drop(columns=[config.DP_TARGET_COL])
    y_test = test_enc[config.DP_TARGET_COL]

    clf = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_SEED)
    clf.fit(x_train, y_train)
    return clf.score(x_test, y_test) * 100.0


def compute_variant_metrics(
    variant: str,
    epsilon_value: float | None,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synth_df: pd.DataFrame,
) -> dict[str, object]:
    """Inputs: variant metadata, real Adult train/test, and synthetic dataframe. Outputs: one dashboard row of Direction 3 metrics. Lifecycle stage: Stage 4 — Evaluation. Reference: `evaluation.metrics`, `evaluation.privacy_fairness`, and the Direction 3 subgroup analysis specification."""
    js_value = base_metrics.mean_js_divergence(real_train, synth_df)
    tstr_value = base_metrics.tstr_accuracy(synth_df, real_test)
    mia_value = base_privacy.membership_inference_attack(real_train, synth_df)
    dp_value = base_privacy.demographic_parity(synth_df)

    female_test = real_test[real_test[config.DP_SENSITIVE_COL] == "Female"].reset_index(drop=True)
    high_income_test = real_test[real_test[config.DP_TARGET_COL] == config.DP_MINORITY_CLASS].reset_index(drop=True)

    tstr_female = subgroup_tstr_accuracy(synth_df, female_test)
    tstr_high_income = subgroup_tstr_accuracy(synth_df, high_income_test)
    female_degradation = tstr_value - tstr_female
    high_income_degradation = tstr_value - tstr_high_income

    score_values = derive_triangle_scores(tstr_value, mia_value, dp_value)
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


def compute_dashboard() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inputs: none. Outputs: dashboard dataframe and subgroup dataframe. Lifecycle stage: Stage 4 — Evaluation. Reference: Direction 3 metric specification for Adult data."""
    real_train = pd.read_csv(config.DATA_DIR / "adult_train.csv")
    real_test = pd.read_csv(config.DATA_DIR / "adult_test.csv")

    rows = []
    for variant in VARIANT_ORDER:
        synth_df = pd.read_csv(synthetic_path(variant))
        rows.append(
            compute_variant_metrics(
                variant=variant,
                epsilon_value=config.DP_EPSILON_VALUES[variant],
                real_train=real_train,
                real_test=real_test,
                synth_df=synth_df,
            )
        )

    dashboard_df = pd.DataFrame(rows)
    subgroup_df = dashboard_df[
        [
            "variant",
            "TSTR",
            "TSTR_Female",
            "TSTR_High_Income",
            "Female_Degradation",
            "High_Income_Degradation",
        ]
    ].rename(
        columns={
            "TSTR": "tstr_overall",
            "TSTR_Female": "tstr_female",
            "TSTR_High_Income": "tstr_high_income",
            "Female_Degradation": "female_degradation",
            "High_Income_Degradation": "high_income_degradation",
        }
    )
    return dashboard_df, subgroup_df


def main(argv: list[str] | None = None) -> int:
    """Inputs: optional CLI argv. Outputs: integer exit code after caching or evaluation. Lifecycle stage: Stage 4 — Evaluation. Reference: Direction 3 orchestration contract."""
    del argv
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    path_map = results_paths()
    if path_map["dashboard"].exists() and not path_map["dry_run_marker"].exists():
        print(f"[CACHE HIT] Skipping evaluation — {path_map['dashboard'].name} already exists")
        return 0

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dashboard_df, subgroup_df = compute_dashboard()
    dashboard_df.to_csv(path_map["dashboard"], index=False)
    subgroup_df.to_csv(path_map["subgroup"], index=False)
    print(dashboard_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
