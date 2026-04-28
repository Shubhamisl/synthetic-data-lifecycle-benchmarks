from __future__ import annotations

import pandas as pd

from benchmarks.generate_benchmark_report import build_summary_tables


def test_build_summary_tables_mentions_best_model_from_three_model_run():
    summary_rows = []
    for dataset in ["adult", "bank", "covertype", "diabetes"]:
        summary_rows.extend(
            [
                {
                    "Dataset": dataset,
                    "Domain": "Domain",
                    "TSTR_Real_Baseline": 90.0,
                    "TSTR_Accuracy": 85.0,
                    "JS_Divergence": 0.12,
                    "MIA_Advantage": 0.18,
                    "Demographic_Parity": 0.22,
                    "Model": "CTGAN",
                },
                {
                    "Dataset": dataset,
                    "Domain": "Domain",
                    "TSTR_Real_Baseline": 90.0,
                    "TSTR_Accuracy": 84.0,
                    "JS_Divergence": 0.10,
                    "MIA_Advantage": 0.16,
                    "Demographic_Parity": 0.20,
                    "Model": "TVAE",
                },
                {
                    "Dataset": dataset,
                    "Domain": "Domain",
                    "TSTR_Real_Baseline": 90.0,
                    "TSTR_Accuracy": 87.0,
                    "JS_Divergence": 0.08,
                    "MIA_Advantage": 0.14,
                    "Demographic_Parity": 0.18,
                    "Model": "TABDDPM",
                },
            ]
        )

    summary_df = pd.DataFrame(summary_rows)
    mean_rank_df = pd.DataFrame(
        [
            {
                "Model": "CTGAN",
                "Mean_TSTR_Rank": 2.0,
                "Mean_JS_Rank": 2.0,
                "Mean_MIA_Rank": 2.0,
                "Mean_DP_Rank": 2.0,
                "Overall_Mean_Rank": 2.0,
            },
            {
                "Model": "TVAE",
                "Mean_TSTR_Rank": 3.0,
                "Mean_JS_Rank": 3.0,
                "Mean_MIA_Rank": 3.0,
                "Mean_DP_Rank": 3.0,
                "Overall_Mean_Rank": 3.0,
            },
            {
                "Model": "TABDDPM",
                "Mean_TSTR_Rank": 1.0,
                "Mean_JS_Rank": 1.0,
                "Mean_MIA_Rank": 1.0,
                "Mean_DP_Rank": 1.0,
                "Overall_Mean_Rank": 1.0,
            },
        ]
    )

    compact_table, detailed_table, dataset_findings, insight_bullets, conclusion = build_summary_tables(
        summary_df, mean_rank_df
    )

    assert "TABDDPM" in " ".join(insight_bullets)
    assert "TABDDPM" in conclusion
    assert set(compact_table["Best Synthetic TSTR (%)"].str.contains("TABDDPM")) == {True}
    assert len(detailed_table) == 12
    assert any("TABDDPM" in finding for finding in dataset_findings)
