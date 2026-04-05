"""Train cache-aware DP-CTGAN variants for Direction 3."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
from dp_triangle.common import dataset_input_paths, dataset_result_dir, get_dataset_spec
from dp_triangle.dp_ctgan import DPCTGANSynthesizer, print_device_banner
from evaluation.metrics import tstr_accuracy

torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)

EXPECTED_MINUTES = {
    "no_dp": {"cuda": 8, "cpu": 20},
    "eps_10": {"cuda": 10, "cpu": 25},
    "eps_1": {"cuda": 12, "cpu": 30},
    "eps_0_5": {"cuda": 15, "cpu": 35},
    "eps_0_1": {"cuda": 20, "cpu": 45},
}


def synthetic_path(key: str, dataset_name: str = "adult") -> Path:
    """
    Build the synthetic CSV path for a dataset variant.

    Inputs: variant key and dataset name.
    Outputs: synthetic CSV path.
    Lifecycle stage: Stage 1 - Training.
    Reference: Direction 3 output specification generalized to supporting datasets.
    """
    return dataset_result_dir(dataset_name) / f"dp_synthetic_{key}.csv"


def model_path(key: str, dataset_name: str = "adult") -> Path:
    """
    Build the saved model path for a dataset variant.

    Inputs: variant key and dataset name.
    Outputs: model pickle path.
    Lifecycle stage: Stage 1 - Training.
    Reference: Direction 3 output specification generalized to supporting datasets.
    """
    if dataset_name == "adult":
        return config.MODEL_SAVE_DIR / f"dp_ctgan_{key}.pkl"
    return dataset_result_dir(dataset_name) / "models" / f"dp_ctgan_{key}.pkl"


def log_path(key: str, dataset_name: str = "adult") -> Path:
    """
    Build the training-log path for a dataset variant.

    Inputs: variant key and dataset name.
    Outputs: training-log JSON path.
    Lifecycle stage: Stage 1 - Training.
    Reference: Direction 3 output specification generalized to supporting datasets.
    """
    return dataset_result_dir(dataset_name) / f"dp_training_log_{key}.json"


def cached_variant_keys(dataset_name: str = "adult") -> list[str]:
    """
    Return variant keys whose model and synthetic outputs already exist.

    Inputs: dataset name.
    Outputs: cached variant keys.
    Lifecycle stage: Stage 1 - Training.
    Reference: Direction 3 smart caching requirement.
    """
    return [
        key
        for key in config.DP_EPSILON_VALUES
        if model_path(key, dataset_name=dataset_name).exists()
        and synthetic_path(key, dataset_name=dataset_name).exists()
    ]


def train_single_variant(
    key: str,
    epsilon: float | None,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str = "adult",
) -> dict[str, object]:
    """
    Train one DP-CTGAN epsilon variant for a dataset.

    Inputs: variant key, epsilon, dataset train dataframe, dataset test dataframe, and dataset name.
    Outputs: JSON-serializable training log dictionary.
    Lifecycle stage: Stage 1 - Training.
    Reference: Direction 3 multi-epsilon DP-CTGAN experiment generalized to supporting datasets.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = get_dataset_spec(dataset_name)
    print_device_banner(device)
    device_key = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ETA] {key} -> ~{EXPECTED_MINUTES[key][device_key]} min {device_key.upper()}")

    if model_path(key, dataset_name=dataset_name).exists() and synthetic_path(key, dataset_name=dataset_name).exists():
        print(f"[CACHE HIT] Skipping {key} - outputs already exist")
        return {
            "variant": key,
            "dataset": dataset_name,
            "epsilon_target": epsilon,
            "epsilon_actual": None,
            "delta": config.DP_TARGET_DELTA,
            "epochs_completed": 0,
            "diverged": False,
            "final_g_loss": None,
            "final_d_loss": None,
            "device_used": device_key,
            "training_time_seconds": 0.0,
        }

    start = time.time()
    model_output = model_path(key, dataset_name=dataset_name)
    synthetic_output = synthetic_path(key, dataset_name=dataset_name)
    log_output = log_path(key, dataset_name=dataset_name)
    synthesizer = DPCTGANSynthesizer(
        epsilon=epsilon,
        epochs=config.DP_EPOCHS,
        batch_size=config.DP_BATCH_SIZE,
        noise_dim=config.DP_NOISE_DIM,
        target_delta=config.DP_TARGET_DELTA,
        max_grad_norm=config.DP_MAX_GRAD_NORM,
        random_seed=config.RANDOM_SEED,
        target_col=spec.target_col,
        sensitive_col=spec.sensitive_attr,
        metadata_dir=model_output.parent,
    )
    synthesizer.fit(train_df)
    synthetic_df = synthesizer.sample(config.DP_N_SYNTHETIC)
    synthetic_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    log_output.parent.mkdir(parents=True, exist_ok=True)

    synthetic_df.to_csv(synthetic_output, index=False)
    synthesizer.save(model_output)

    training_log = {
        "variant": key,
        "dataset": dataset_name,
        "epsilon_target": epsilon,
        "epsilon_actual": synthesizer.epsilon_actual,
        "delta": config.DP_TARGET_DELTA,
        "epochs_completed": synthesizer.epochs_completed,
        "diverged": synthesizer.diverged,
        "final_g_loss": synthesizer.final_g_loss,
        "final_d_loss": synthesizer.final_d_loss,
        "device_used": device_key,
        "training_time_seconds": time.time() - start,
    }
    log_output.write_text(json.dumps(training_log, indent=2), encoding="utf-8")

    if key == "no_dp" and dataset_name == "adult":
        tstr_value = tstr_accuracy(synthetic_df, test_df)
        if abs(tstr_value - 81.43) > 5.0:
            print(
                f"WARNING: no_dp TSTR={tstr_value:.2f}% deviates >5% from expected "
                "81.43% - check preprocessing or random seed"
            )
    return training_log


def main(argv: list[str] | None = None) -> int:
    """
    Train all configured epsilon variants for one dataset.

    Inputs: optional CLI argv where the first argument is the dataset name.
    Outputs: integer exit code after variant training.
    Lifecycle stage: Stage 1 - Training.
    Reference: Direction 3 orchestration contract generalized to multiple datasets.
    """
    dataset_name = argv[0] if argv else "adult"
    paths = dataset_input_paths(dataset_name)
    train_df = pd.read_csv(paths["train"])
    test_df = pd.read_csv(paths["test"])
    for key, epsilon in config.DP_EPSILON_VALUES.items():
        train_single_variant(key, epsilon, train_df, test_df, dataset_name=dataset_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
