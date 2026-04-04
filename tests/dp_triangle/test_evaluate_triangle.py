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
