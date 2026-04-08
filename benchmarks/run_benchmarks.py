from __future__ import annotations

import subprocess
import sys
import time

import pandas as pd

from . import benchmark_models
from .common import BENCHMARK_ROOT, get_dataset_paths
from .visualize_benchmarks import MEAN_RANK_PATH, SUMMARY_PATH, required_plot_paths


def _configure_stdout_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _run_module(module_name: str) -> None:
    subprocess.run([sys.executable, "-m", module_name], check=True)


def _benchmark_model_ids() -> tuple[str, ...]:
    return benchmark_models.get_trainable_benchmark_model_ids()


def _all_exist(paths) -> bool:
    return all(path.exists() for path in paths)


def _dataset_files() -> list:
    paths = []
    for dataset in ("adult", "bank", "covertype", "diabetes"):
        dataset_paths = get_dataset_paths(dataset)
        paths.extend([dataset_paths["train"], dataset_paths["test"]])
    return paths


def _synthetic_files() -> list:
    paths = []
    model_ids = _benchmark_model_ids()
    for dataset in ("bank", "covertype", "diabetes"):
        dataset_paths = get_dataset_paths(dataset)
        paths.extend([dataset_paths[model_id] for model_id in model_ids])
    return paths


def main() -> None:
    _configure_stdout_utf8()
    model_ids = _benchmark_model_ids()
    print("=" * 65)
    print("   CROSS-DOMAIN BENCHMARKING PIPELINE")
    print(f"   4 Datasets x {len(model_ids)} Models x 4 Metrics")
    print("=" * 65)

    start = time.time()

    if _all_exist(_dataset_files()):
        print("  [ok] All datasets ready - skipping download")
    else:
        _run_module("benchmarks.download_datasets")

    for dataset in ("bank", "covertype", "diabetes"):
        dataset_paths = get_dataset_paths(dataset)
        if all(dataset_paths[model_id].exists() for model_id in model_ids):
            print(f"  [ok] {dataset} synthetic data exists - skipping")
        else:
            _run_module("benchmarks.train_benchmark_models")
            break

    if SUMMARY_PATH.exists():
        print("  [ok] Evaluation complete - skipping")
    else:
        _run_module("benchmarks.evaluate_benchmarks")

    if _all_exist(required_plot_paths()):
        print("  [ok] All plots exist - skipping")
    else:
        _run_module("benchmarks.visualize_benchmarks")

    print(pd.read_csv(SUMMARY_PATH).to_string(index=False))
    print(pd.read_csv(MEAN_RANK_PATH).to_string(index=False))
    print(f"Elapsed time: {time.time() - start:.1f} seconds")
    print("=" * 65)
    print("   BENCHMARKING COMPLETE")
    print(f"   Results in {BENCHMARK_ROOT / 'results'}")
    print(f"   Plots   in {BENCHMARK_ROOT / 'plots'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
