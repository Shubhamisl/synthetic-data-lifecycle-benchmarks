from __future__ import annotations

from pathlib import Path

import pandas as pd

import config


def test_dry_run_generates_dummy_outputs_and_dashboard(tmp_path, monkeypatch):
    from dp_triangle import run_direction3

    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    models_dir = tmp_path / "models" / "saved"
    data_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    source_train = pd.read_csv(config.DATA_DIR / "adult_train.csv")
    source_test = pd.read_csv(config.DATA_DIR / "adult_test.csv")
    source_train.to_csv(data_dir / "adult_train.csv", index=False)
    source_test.to_csv(data_dir / "adult_test.csv", index=False)

    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(config, "MODEL_SAVE_DIR", models_dir)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)

    exit_code = run_direction3.main(["--dry-run"])

    assert exit_code == 0
    for key in config.DP_EPSILON_VALUES:
        assert (results_dir / f"dp_synthetic_{key}.csv").exists()
    assert (results_dir / "dp_triangle_dashboard.csv").exists()
    assert (results_dir / "dp_subgroup_fairness.csv").exists()
    assert (results_dir / "figure4_epsilon_tradeoff_curve.png").exists()
    assert (results_dir / "figure5_pff_radar_chart.png").exists()
    assert (results_dir / "figure6_intersectional_fairness.png").exists()


def test_dry_run_requires_expected_adult_shapes(tmp_path, monkeypatch):
    from dp_triangle import run_direction3

    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    models_dir = tmp_path / "models" / "saved"
    data_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    pd.DataFrame({"income": [0], "sex": ["Male"]}).to_csv(data_dir / "adult_train.csv", index=False)
    pd.DataFrame({"income": [0], "sex": ["Male"]}).to_csv(data_dir / "adult_test.csv", index=False)

    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(config, "MODEL_SAVE_DIR", models_dir)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)

    try:
        run_direction3.main(["--dry-run"])
    except ValueError as exc:
        assert "expected shape" in str(exc)
    else:
        raise AssertionError("Expected dry-run shape validation to fail")


def test_triangle_score_column_prefers_adjusted_metric():
    from dp_triangle.run_direction3 import triangle_score_column

    dashboard = pd.DataFrame(
        {
            "Triangle_Score": [0.70, 0.75],
            "Triangle_Score_Adjusted": [0.70, 0.00],
        }
    )

    assert triangle_score_column(dashboard) == "Triangle_Score_Adjusted"


def test_triangle_score_column_falls_back_to_raw_metric():
    from dp_triangle.run_direction3 import triangle_score_column

    dashboard = pd.DataFrame({"Triangle_Score": [0.70, 0.75]})

    assert triangle_score_column(dashboard) == "Triangle_Score"
