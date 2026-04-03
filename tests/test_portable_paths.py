from pathlib import Path

import config
import evaluation.metrics as metrics
import evaluation.privacy_fairness as privacy_fairness
import models.train_models as train_models
import results.visualize as results_visualize


def test_config_paths_are_repo_relative_path_objects():
    assert isinstance(config.PROJECT_ROOT, Path)
    assert isinstance(config.DATA_DIR, Path)
    assert isinstance(config.RESULTS_DIR, Path)
    assert isinstance(config.MODEL_SAVE_DIR, Path)
    assert config.DATA_DIR == config.PROJECT_ROOT / "data"
    assert config.RESULTS_DIR == config.PROJECT_ROOT / "results"
    assert config.MODEL_SAVE_DIR == config.PROJECT_ROOT / "models" / "saved"


def test_modules_use_config_derived_paths():
    assert metrics.TRAIN_PATH == config.DATA_DIR / "adult_train.csv"
    assert metrics.TEST_PATH == config.DATA_DIR / "adult_test.csv"
    assert metrics.CTGAN_PATH == config.RESULTS_DIR / "ctgan_synthetic.csv"
    assert metrics.TVAE_PATH == config.RESULTS_DIR / "tvae_synthetic.csv"
    assert privacy_fairness.TRAIN_PATH == config.DATA_DIR / "adult_train.csv"
    assert privacy_fairness.RESULTS_DIR == config.RESULTS_DIR
    assert train_models.TRAIN_PATH == config.DATA_DIR / "adult_train.csv"
    assert train_models.RESULTS_DIR == config.RESULTS_DIR
    assert results_visualize.RESULTS_DIR == config.RESULTS_DIR
