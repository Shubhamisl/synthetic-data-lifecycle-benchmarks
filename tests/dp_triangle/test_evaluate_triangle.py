from __future__ import annotations

import math

import pytest

import config


def test_config_exposes_direction3_hyperparameters():
    assert config.DP_EPSILON_VALUES == {
        "no_dp": None,
        "eps_10": 10.0,
        "eps_1": 1.0,
        "eps_0_5": 0.5,
        "eps_0_1": 0.1,
    }
    assert config.DP_EPOCHS == 300
    assert config.DP_BATCH_SIZE == 500
    assert config.DP_NOISE_DIM == 128
    assert config.DP_TARGET_COL == "income"
    assert config.DP_SENSITIVE_COL == "sex"


def test_derive_triangle_scores_normalizes_metrics():
    from dp_triangle.evaluate_triangle import derive_triangle_scores

    scores = derive_triangle_scores(
        tstr_accuracy=82.5,
        mia_advantage=0.2,
        demographic_parity=0.15,
    )

    assert scores["Privacy_Score"] == pytest.approx(0.8)
    assert scores["Utility_Score"] == pytest.approx(0.825)
    assert scores["Fairness_Score"] == pytest.approx(0.85)
    assert scores["Triangle_Score"] == pytest.approx((0.8 + 0.825 + 0.85) / 3.0)


def test_derive_triangle_scores_preserves_nan_fairness():
    from dp_triangle.evaluate_triangle import derive_triangle_scores

    scores = derive_triangle_scores(
        tstr_accuracy=81.0,
        mia_advantage=0.3,
        demographic_parity=math.nan,
    )

    assert scores["Privacy_Score"] == pytest.approx(0.7)
    assert scores["Utility_Score"] == pytest.approx(0.81)
    assert math.isnan(scores["Fairness_Score"])
    assert math.isnan(scores["Triangle_Score"])


def test_derive_triangle_scores_penalizes_collapsed_positive_class():
    from dp_triangle.evaluate_triangle import derive_triangle_scores

    scores = derive_triangle_scores(
        tstr_accuracy=75.0,
        mia_advantage=0.5,
        demographic_parity=0.0,
        synthetic_positive_rate=0.0,
        real_positive_rate=0.25,
    )

    assert scores["Synthetic_Positive_Rate"] == pytest.approx(0.0)
    assert scores["Positive_Class_Retention"] == pytest.approx(0.0)
    assert scores["Collapsed_Minority_Class"] is True
    assert scores["Triangle_Score"] == pytest.approx((0.5 + 0.75 + 1.0) / 3.0)
    assert scores["Triangle_Score_Adjusted"] == pytest.approx(0.0)
    assert scores["Collapse_Reason"] == "positive_class_missing"


def test_results_paths_default_to_adult_results_dir():
    from dp_triangle.evaluate_triangle import results_paths

    paths = results_paths("adult")

    assert paths["dashboard"] == config.RESULTS_DIR / "dp_triangle_dashboard.csv"
    assert paths["subgroup"] == config.RESULTS_DIR / "dp_subgroup_fairness.csv"


def test_results_paths_scope_supporting_datasets_under_benchmarks():
    from dp_triangle.evaluate_triangle import results_paths

    bank_root = config.PROJECT_ROOT / "benchmarks" / "results" / "dp_triangle" / "bank"
    paths = results_paths("bank")

    assert paths["dashboard"] == bank_root / "dp_triangle_dashboard.csv"
    assert paths["subgroup"] == bank_root / "dp_subgroup_fairness.csv"


def test_compute_generic_variant_metrics_uses_generic_subgroup_columns(monkeypatch):
    import pandas as pd

    from dp_triangle.common import get_dataset_spec
    from dp_triangle.evaluate_triangle import compute_generic_variant_metrics

    spec = get_dataset_spec("bank")
    real_train = pd.DataFrame(
        {
            "balance": [1, 2, 3, 4],
            "target": [0, 1, 0, 1],
            "age_group": ["young", "older", "young", "older"],
        }
    )
    real_test = real_train.copy()
    synth_df = real_train.copy()

    monkeypatch.setattr("dp_triangle.evaluate_triangle.benchmark_mean_js_divergence", lambda *args, **kwargs: 0.12)
    monkeypatch.setattr("dp_triangle.evaluate_triangle.benchmark_tstr_accuracy", lambda *args, **kwargs: 80.0)
    monkeypatch.setattr("dp_triangle.evaluate_triangle.benchmark_membership_inference_advantage", lambda *args, **kwargs: 0.2)
    monkeypatch.setattr("dp_triangle.evaluate_triangle.numeric_feature_columns", lambda *args, **kwargs: ["balance"])

    row = compute_generic_variant_metrics(spec, "eps_1", 1.0, real_train, real_test, synth_df)

    assert row["Demo_Parity"] == pytest.approx(1.0)
    assert row["TSTR_Sensitive_Subgroup"] == pytest.approx(80.0)
    assert row["TSTR_Positive_Class"] == pytest.approx(80.0)
    assert row["Sensitive_Subgroup_Degradation"] == pytest.approx(0.0)
    assert row["Positive_Class_Degradation"] == pytest.approx(0.0)
    assert row["Sensitive_Subgroup_Label"] == "young"


def test_compute_generic_variant_metrics_handles_privacy_utility_only_dataset(monkeypatch):
    import pandas as pd

    from dp_triangle.common import get_dataset_spec
    from dp_triangle.evaluate_triangle import compute_generic_variant_metrics

    spec = get_dataset_spec("covertype")
    real_train = pd.DataFrame({"Elevation": [1, 2, 3], "Cover_Type": [1, 2, 1]})
    real_test = real_train.copy()
    synth_df = real_train.copy()

    monkeypatch.setattr("dp_triangle.evaluate_triangle.benchmark_mean_js_divergence", lambda *args, **kwargs: 0.05)
    monkeypatch.setattr("dp_triangle.evaluate_triangle.benchmark_tstr_accuracy", lambda *args, **kwargs: 70.0)
    monkeypatch.setattr("dp_triangle.evaluate_triangle.benchmark_membership_inference_advantage", lambda *args, **kwargs: 0.25)
    monkeypatch.setattr("dp_triangle.evaluate_triangle.numeric_feature_columns", lambda *args, **kwargs: ["Elevation"])

    row = compute_generic_variant_metrics(spec, "eps_1", 1.0, real_train, real_test, synth_df)

    assert math.isnan(row["Demo_Parity"])
    assert math.isnan(row["Fairness_Score"])
    assert math.isnan(row["Triangle_Score"])
    assert math.isnan(row["TSTR_Sensitive_Subgroup"])
    assert row["TSTR_Positive_Class"] == pytest.approx(70.0)
