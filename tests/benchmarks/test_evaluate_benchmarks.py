from __future__ import annotations

import math

import pandas as pd

from benchmarks import benchmark_models
from benchmarks.evaluate_benchmarks import (
    demographic_parity_difference,
    mean_js_divergence,
    rank_models,
    _evaluation_model_specs,
)


def test_demographic_parity_difference_uses_positive_rate_gap():
    df = pd.DataFrame(
        {
            "age_group": ["young", "young", "older", "older"],
            "target": [1, 0, 1, 1],
        }
    )

    assert demographic_parity_difference(df, "age_group", "target") == 0.5


def test_mean_js_divergence_is_zero_for_identical_numeric_columns():
    real_df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "target": [0, 1, 0]})
    synth_df = real_df.copy()

    assert math.isclose(mean_js_divergence(real_df, synth_df, ["x"]), 0.0, abs_tol=1e-6)


def test_rank_models_prefers_higher_tstr_and_lower_js():
    summary_df = pd.DataFrame(
        [
            {
                "Dataset": "bank",
                "Model": "CTGAN",
                "TSTR_Accuracy": 81.0,
                "JS_Divergence": 0.10,
                "MIA_Advantage": 0.11,
                "Demographic_Parity": 0.20,
            },
            {
                "Dataset": "bank",
                "Model": "TVAE",
                "TSTR_Accuracy": 79.0,
                "JS_Divergence": 0.12,
                "MIA_Advantage": 0.15,
                "Demographic_Parity": 0.25,
            },
        ]
    )

    ranked = rank_models(summary_df)

    ctgan_rank = ranked.loc[ranked["Model"] == "CTGAN", "Overall_Mean_Rank"].iloc[0]
    tvae_rank = ranked.loc[ranked["Model"] == "TVAE", "Overall_Mean_Rank"].iloc[0]
    assert ctgan_rank < tvae_rank


def test_rank_models_handles_three_models():
    summary_df = pd.DataFrame(
        [
            {
                "Dataset": "bank",
                "Model": "CTGAN",
                "TSTR_Accuracy": 81.0,
                "JS_Divergence": 0.10,
                "MIA_Advantage": 0.11,
                "Demographic_Parity": 0.20,
            },
            {
                "Dataset": "bank",
                "Model": "TVAE",
                "TSTR_Accuracy": 79.0,
                "JS_Divergence": 0.12,
                "MIA_Advantage": 0.15,
                "Demographic_Parity": 0.25,
            },
            {
                "Dataset": "bank",
                "Model": "TABDDPM",
                "TSTR_Accuracy": 83.0,
                "JS_Divergence": 0.08,
                "MIA_Advantage": 0.09,
                "Demographic_Parity": 0.18,
            },
        ]
    )

    ranked = rank_models(summary_df)

    assert ranked["Model"].tolist() == ["CTGAN", "TABDDPM", "TVAE"]
    assert ranked.loc[ranked["Model"] == "TABDDPM", "Overall_Mean_Rank"].iloc[0] < ranked.loc[
        ranked["Model"] == "CTGAN", "Overall_Mean_Rank"
    ].iloc[0]


def test_evaluation_model_specs_follow_registry_display_names(monkeypatch):
    monkeypatch.setattr(
        benchmark_models,
        "BENCHMARK_MODELS",
        {
            "ctgan": benchmark_models.BenchmarkModelSpec("ctgan", "CTGAN"),
            "tvae": benchmark_models.BenchmarkModelSpec("tvae", "TVAE"),
            "tabddpm": benchmark_models.BenchmarkModelSpec("tabddpm", "TABDDPM"),
        },
    )

    specs = _evaluation_model_specs()

    assert [spec.model_id for spec in specs] == ["ctgan", "tvae", "tabddpm"]
    assert [spec.display_name for spec in specs] == ["CTGAN", "TVAE", "TABDDPM"]
