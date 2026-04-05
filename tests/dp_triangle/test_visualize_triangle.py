from __future__ import annotations

import pandas as pd

import config


def test_triangle_score_column_prefers_adjusted_metric_for_visuals():
    from dp_triangle.visualize_triangle import triangle_score_column

    dashboard = pd.DataFrame(
        {
            "Triangle_Score": [0.70, 0.75],
            "Triangle_Score_Adjusted": [0.70, 0.00],
        }
    )

    assert triangle_score_column(dashboard) == "Triangle_Score_Adjusted"


def test_variant_radar_label_marks_collapsed_variants():
    from dp_triangle.visualize_triangle import variant_radar_label

    row = pd.Series(
        {
            "Triangle_Score": 0.751,
            "Triangle_Score_Adjusted": 0.0,
            "Collapsed_Minority_Class": True,
        }
    )

    label = variant_radar_label("eps_0_1", row, "Triangle_Score_Adjusted")

    assert "epsilon=0.1" in label
    assert "collapsed" in label.lower()
    assert "0.00" in label


def test_display_variant_label_marks_collapsed_variants():
    from dp_triangle.visualize_triangle import display_variant_label

    row = pd.Series({"Collapsed_Minority_Class": True})

    assert display_variant_label("eps_0_1", row) == "epsilon=0.1 *"


def test_result_paths_scope_supporting_dataset_figures_under_benchmarks():
    from dp_triangle.visualize_triangle import result_paths

    bank_root = config.PROJECT_ROOT / "benchmarks" / "results" / "dp_triangle" / "bank"
    paths = result_paths("bank")

    assert paths["dashboard"] == bank_root / "dp_triangle_dashboard.csv"
    assert paths["fig4"] == bank_root / "figure4_epsilon_tradeoff_curve.png"
