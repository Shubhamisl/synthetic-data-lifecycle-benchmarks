from __future__ import annotations

import config


def test_output_path_defaults_to_adult_results_dir():
    from dp_triangle.post_hoc_debiasing import output_path

    assert output_path("adult") == config.RESULTS_DIR / "dp_post_hoc_debiasing.csv"


def test_output_path_scopes_supporting_dataset_under_benchmarks():
    from dp_triangle.post_hoc_debiasing import output_path

    bank_root = config.PROJECT_ROOT / "benchmarks" / "results" / "dp_triangle" / "bank"
    assert output_path("bank") == bank_root / "dp_post_hoc_debiasing.csv"
