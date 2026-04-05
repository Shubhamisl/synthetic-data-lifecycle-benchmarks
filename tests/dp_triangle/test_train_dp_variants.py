from __future__ import annotations

import config


def test_training_paths_default_to_adult_results_and_models():
    from dp_triangle import train_dp_variants

    assert train_dp_variants.synthetic_path("eps_1", dataset_name="adult") == config.RESULTS_DIR / "dp_synthetic_eps_1.csv"
    assert train_dp_variants.model_path("eps_1", dataset_name="adult") == config.MODEL_SAVE_DIR / "dp_ctgan_eps_1.pkl"
    assert train_dp_variants.log_path("eps_1", dataset_name="adult") == config.RESULTS_DIR / "dp_training_log_eps_1.json"


def test_training_paths_scope_supporting_datasets_under_benchmarks():
    from dp_triangle import train_dp_variants

    bank_root = config.PROJECT_ROOT / "benchmarks" / "results" / "dp_triangle" / "bank"

    assert train_dp_variants.synthetic_path("eps_1", dataset_name="bank") == bank_root / "dp_synthetic_eps_1.csv"
    assert train_dp_variants.model_path("eps_1", dataset_name="bank") == bank_root / "models" / "dp_ctgan_eps_1.pkl"
    assert train_dp_variants.log_path("eps_1", dataset_name="bank") == bank_root / "dp_training_log_eps_1.json"
