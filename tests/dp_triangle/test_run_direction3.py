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


def test_clear_direction3_result_artifacts_preserves_models_and_synthetic(tmp_path, monkeypatch):
    from dp_triangle import run_direction3

    results_dir = tmp_path / "results"
    model_dir = tmp_path / "models" / "saved"
    results_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    removable = [
        "dp_triangle_dashboard.csv",
        "dp_subgroup_fairness.csv",
        "dp_post_hoc_debiasing.csv",
        "figure4_epsilon_tradeoff_curve.png",
        "figure5_pff_radar_chart.png",
        "figure6_intersectional_fairness.png",
        "figure7_post_hoc_recovery.png",
        "dp_direction3_dry_run.marker",
    ]
    for name in removable:
        (results_dir / name).write_text("x", encoding="utf-8")

    preserved_synthetic = results_dir / "dp_synthetic_eps_1.csv"
    preserved_model = model_dir / "dp_ctgan_eps_1.pkl"
    preserved_synthetic.write_text("synthetic", encoding="utf-8")
    preserved_model.write_text("model", encoding="utf-8")

    monkeypatch.setattr(config, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(config, "MODEL_SAVE_DIR", model_dir)

    run_direction3.clear_direction3_result_artifacts()

    for name in removable:
        assert not (results_dir / name).exists()
    assert preserved_synthetic.exists()
    assert preserved_model.exists()


def test_parse_args_accepts_dataset_selection():
    from dp_triangle.run_direction3 import parse_args

    args = parse_args(["--dataset", "bank", "--refresh-results"])

    assert args.dataset == "bank"
    assert args.refresh_results is True


def test_clear_direction3_result_artifacts_uses_dataset_scope_for_supporting_dataset(tmp_path, monkeypatch):
    from dp_triangle import run_direction3

    project_root = tmp_path
    bank_results = project_root / "benchmarks" / "results" / "dp_triangle" / "bank"
    bank_models = bank_results / "models"
    bank_results.mkdir(parents=True)
    bank_models.mkdir(parents=True)

    removable = [
        "dp_triangle_dashboard.csv",
        "dp_subgroup_fairness.csv",
        "dp_post_hoc_debiasing.csv",
        "figure4_epsilon_tradeoff_curve.png",
        "figure5_pff_radar_chart.png",
        "figure6_intersectional_fairness.png",
        "figure7_post_hoc_recovery.png",
        "dp_direction3_dry_run.marker",
    ]
    for name in removable:
        (bank_results / name).write_text("x", encoding="utf-8")

    preserved_synthetic = bank_results / "dp_synthetic_eps_1.csv"
    preserved_model = bank_models / "dp_ctgan_eps_1.pkl"
    preserved_synthetic.write_text("synthetic", encoding="utf-8")
    preserved_model.write_text("model", encoding="utf-8")

    monkeypatch.setattr(config, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(config, "RESULTS_DIR", project_root / "results")
    monkeypatch.setattr(config, "MODEL_SAVE_DIR", project_root / "models" / "saved")

    run_direction3.clear_direction3_result_artifacts("bank")

    for name in removable:
        assert not (bank_results / name).exists()
    assert preserved_synthetic.exists()
    assert preserved_model.exists()


def test_bank_dry_run_generates_dataset_scoped_outputs(tmp_path, monkeypatch):
    from dp_triangle import run_direction3

    benchmarks_dir = tmp_path / "benchmarks" / "datasets"
    benchmarks_dir.mkdir(parents=True)
    bank_train = pd.DataFrame(
        {
            "balance": [10, 20, 30, 40, 50, 60],
            "target": [0, 1, 0, 1, 0, 1],
            "age_group": ["young", "older", "young", "older", "young", "older"],
        }
    )
    bank_test = bank_train.copy()
    bank_train.to_csv(benchmarks_dir / "bank_train.csv", index=False)
    bank_test.to_csv(benchmarks_dir / "bank_test.csv", index=False)

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "MODEL_SAVE_DIR", tmp_path / "models" / "saved")

    exit_code = run_direction3.main(["--dataset", "bank", "--dry-run"])

    assert exit_code == 0
    result_root = tmp_path / "benchmarks" / "results" / "dp_triangle" / "bank"
    assert (result_root / "dp_triangle_dashboard.csv").exists()
    assert (result_root / "dp_subgroup_fairness.csv").exists()
    assert (result_root / "figure4_epsilon_tradeoff_curve.png").exists()
    assert (result_root / "figure5_pff_radar_chart.png").exists()
    assert (result_root / "figure6_intersectional_fairness.png").exists()


def test_covertype_dry_run_generates_privacy_utility_only_outputs(tmp_path, monkeypatch):
    from dp_triangle import run_direction3

    benchmarks_dir = tmp_path / "benchmarks" / "datasets"
    benchmarks_dir.mkdir(parents=True)
    covertype_train = pd.DataFrame(
        {
            "Elevation": [1, 2, 3, 4, 5, 6, 7],
            "Slope": [1, 2, 3, 4, 5, 6, 7],
            "Cover_Type": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    covertype_test = covertype_train.copy()
    covertype_train.to_csv(benchmarks_dir / "covertype_train.csv", index=False)
    covertype_test.to_csv(benchmarks_dir / "covertype_test.csv", index=False)

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "MODEL_SAVE_DIR", tmp_path / "models" / "saved")

    exit_code = run_direction3.main(["--dataset", "covertype", "--dry-run"])

    assert exit_code == 0
    result_root = tmp_path / "benchmarks" / "results" / "dp_triangle" / "covertype"
    assert (result_root / "dp_triangle_dashboard.csv").exists()
    assert not (result_root / "dp_subgroup_fairness.csv").exists()
    assert (result_root / "figure4_epsilon_tradeoff_curve.png").exists()
    assert not (result_root / "figure5_pff_radar_chart.png").exists()
    assert not (result_root / "figure6_intersectional_fairness.png").exists()
