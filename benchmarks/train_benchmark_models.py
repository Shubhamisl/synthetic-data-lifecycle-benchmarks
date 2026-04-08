from __future__ import annotations

import shutil
import sys
from typing import Callable

import pandas as pd

from . import benchmark_models
from .common import (
    BENCHMARK_ROOT,
    DATASET_REGISTRY,
    ensure_benchmark_dirs,
    get_dataset_paths,
    log_benchmark_event,
    validate_dataframe_schema,
    validate_required_non_null_columns,
    validate_target_values,
)

SYNTHETIC_ROW_COUNT = 10_000
CTGAN_EPOCHS = 300
CTGAN_BATCH_SIZE = 500
TVAE_EPOCHS = 300


def _benchmark_model_ids() -> tuple[str, ...]:
    return benchmark_models.get_trainable_benchmark_model_ids()


def detect_ctgan_discrete_columns(dataset_name: str, train_df: pd.DataFrame) -> list[str]:
    if dataset_name == "covertype":
        return [
            column
            for column in train_df.columns
            if column == "Cover_Type"
            or column.startswith("Wilderness_Area")
            or column.startswith("Soil_Type")
        ]
    return train_df.select_dtypes(include=["object", "category"]).columns.tolist()


def validate_synthetic_dataset(
    dataset_name: str,
    train_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    target_col: str,
    valid_targets: set[int] | set[str] | None,
    expected_rows: int | None = None,
) -> list[str]:
    if list(train_df.columns) != list(synth_df.columns):
        raise ValueError(f"{dataset_name}: schema mismatch")

    if expected_rows is not None and synth_df.shape != (expected_rows, len(train_df.columns)):
        raise ValueError(
            f"{dataset_name}: synthetic output shape {synth_df.shape} does not match "
            f"expected {(expected_rows, len(train_df.columns))}"
        )

    if synth_df.isna().sum().sum():
        raise ValueError(f"{dataset_name}: synthetic output has NaN values")

    if dataset_name == "covertype":
        target_series = pd.to_numeric(synth_df[target_col], errors="raise")
        if not (target_series == target_series.astype(int)).all():
            raise ValueError(f"{dataset_name}: target values must be integer encoded")

    if valid_targets is not None:
        observed = set(synth_df[target_col].dropna().unique().tolist())
        if not observed.issubset(valid_targets):
            raise ValueError(f"{dataset_name}: invalid target values")

    warnings: list[str] = []
    for column in synth_df.columns:
        if synth_df[column].nunique(dropna=False) == 1:
            warnings.append(f"{dataset_name}: column {column} is constant")

    return warnings


def copy_adult_synthetic_artifacts() -> dict[str, pd.DataFrame]:
    ensure_benchmark_dirs()
    spec = DATASET_REGISTRY["adult"]
    train_df = pd.read_csv(get_dataset_paths("adult")["train"])
    promoted: dict[str, pd.DataFrame] = {}

    for model_id in _benchmark_model_ids():
        model_spec = benchmark_models.get_benchmark_model_spec(model_id)
        if model_id == "tabddpm":
            temp_df = _train_tabddpm_samples("adult", train_df, spec.target_col)
        else:
            if model_spec.adult_source_path is None:
                raise RuntimeError(f"{model_spec.display_name} adult benchmark artifact seam is not configured")

            source_path = model_spec.adult_source_path
            temp_df = pd.read_csv(source_path)
        promoted[model_id], _ = _validate_and_promote_synthetic(
            "adult",
            model_id,
            train_df,
            temp_df,
            spec.target_col,
            spec.valid_target_values,
            expected_rows=None,
        )

    return promoted


def _load_ctgan_class():
    from ctgan import CTGAN

    return CTGAN


def _load_tvae_dependencies():
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import TVAESynthesizer

    return SingleTableMetadata, TVAESynthesizer


def _load_tabddpm_plugin():
    from synthcity.plugins.generic.plugin_ddpm import TabDDPMPlugin

    return TabDDPMPlugin


def _warn(dataset_name: str, message: str) -> None:
    warning_message = f"WARNING [{dataset_name}] {message}"
    print(warning_message)
    log_benchmark_event("training", dataset_name, "WARNING", message)


def _format_target_distribution(df: pd.DataFrame, target_col: str) -> dict[object, int]:
    counts = df[target_col].value_counts(dropna=False).sort_index()
    return counts.to_dict()


def _coerce_covertype_targets(dataset_name: str, synth_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if dataset_name != "covertype" or target_col not in synth_df.columns:
        return synth_df

    coerced = synth_df.copy()
    numeric_target = pd.to_numeric(coerced[target_col], errors="raise")
    if (numeric_target == numeric_target.astype(int)).all():
        coerced[target_col] = numeric_target.astype(int)
    return coerced


def _validate_and_promote_synthetic(
    dataset_name: str,
    model_name: str,
    train_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    target_col: str,
    valid_targets: set[int] | set[str] | None,
    expected_rows: int | None = SYNTHETIC_ROW_COUNT,
) -> tuple[pd.DataFrame, list[str]]:
    synthetic_dir = get_dataset_paths(dataset_name)[model_name.lower()].parent
    final_path = get_dataset_paths(dataset_name)[model_name.lower()]
    temp_path = synthetic_dir / f".{dataset_name}_{model_name.lower()}_tmp.csv"

    try:
        synth_df.to_csv(temp_path, index=False)
        reloaded_df = pd.read_csv(temp_path)
        warnings = _validate_saved_synthetic(
            dataset_name,
            train_df,
            reloaded_df,
            target_col,
            valid_targets,
            expected_rows=expected_rows,
        )
        temp_path.replace(final_path)
        return pd.read_csv(final_path), warnings
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        if final_path.exists():
            final_path.unlink()
        raise


def _print_dataset_completion(
    dataset_name: str,
    target_col: str,
    synth_outputs: dict[str, pd.DataFrame],
) -> None:
    model_status = " | ".join(
        f"{benchmark_models.get_benchmark_model_spec(model_id).display_name} complete"
        for model_id in synth_outputs
    )
    first_output = next(iter(synth_outputs.values()))
    print(f"{dataset_name} - {model_status}")
    print(f"Synthetic shape: {first_output.shape}")
    print(f"Target distribution: {_format_target_distribution(first_output, target_col)}")


def _train_ctgan_samples(dataset_name: str, train_df: pd.DataFrame) -> pd.DataFrame:
    ctgan_class = _load_ctgan_class()
    model = ctgan_class(epochs=CTGAN_EPOCHS, batch_size=CTGAN_BATCH_SIZE, verbose=True)
    model.fit(train_df, discrete_columns=detect_ctgan_discrete_columns(dataset_name, train_df))
    synth_df = model.sample(SYNTHETIC_ROW_COUNT)
    if not isinstance(synth_df, pd.DataFrame):
        synth_df = pd.DataFrame(synth_df, columns=train_df.columns)
    return synth_df.reindex(columns=train_df.columns)


def _train_tvae_samples(train_df: pd.DataFrame) -> pd.DataFrame:
    metadata_class, tvae_class = _load_tvae_dependencies()
    metadata = metadata_class()
    metadata.detect_from_dataframe(train_df)
    model = tvae_class(metadata, epochs=TVAE_EPOCHS)
    model.fit(train_df)
    synth_df = model.sample(num_rows=SYNTHETIC_ROW_COUNT)
    if not isinstance(synth_df, pd.DataFrame):
        synth_df = pd.DataFrame(synth_df, columns=train_df.columns)
    return synth_df.reindex(columns=train_df.columns)


def _train_tabddpm_samples(_dataset_name: str, train_df: pd.DataFrame, _target_col: str) -> pd.DataFrame:
    if not benchmark_models.get_benchmark_model_spec("tabddpm").trainable:
        raise RuntimeError("TABDDPM benchmark adapter requires synthcity with the ddpm plugin installed")

    tabddpm_class = _load_tabddpm_plugin()
    model = tabddpm_class(is_classification=True, n_iter=CTGAN_EPOCHS)
    model.fit(train_df)
    synth_df = model.generate(SYNTHETIC_ROW_COUNT)
    if hasattr(synth_df, "dataframe"):
        synth_df = synth_df.dataframe()
    if not isinstance(synth_df, pd.DataFrame):
        synth_df = pd.DataFrame(synth_df, columns=train_df.columns)
    return synth_df.reindex(columns=train_df.columns)


def _validate_saved_synthetic(
    dataset_name: str,
    train_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    target_col: str,
    valid_targets: set[int] | set[str] | None,
    expected_rows: int | None = SYNTHETIC_ROW_COUNT,
) -> list[str]:
    validate_dataframe_schema(synth_df, dataset_name, required_columns=train_df.columns.tolist())
    validate_target_values(synth_df, target_col, valid_targets, dataset_name)
    return validate_synthetic_dataset(
        dataset_name,
        train_df,
        synth_df,
        target_col,
        valid_targets,
        expected_rows=expected_rows,
    )


def _validate_training_input(dataset_name: str, train_df: pd.DataFrame) -> None:
    validate_dataframe_schema(train_df, dataset_name, required_columns=train_df.columns.tolist())
    if train_df.isna().sum().sum():
        raise ValueError(f"{dataset_name}: training input has NaN values")
    validate_required_non_null_columns(train_df, dataset_name, train_df.columns.tolist())


def _process_training_dataset(
    dataset_name: str,
    train_df: pd.DataFrame,
    target_col: str,
    valid_targets: set[int] | set[str] | None,
    model_ids: tuple[str, ...] | None = None,
    ctgan_sampler: Callable[[str, pd.DataFrame], pd.DataFrame] = _train_ctgan_samples,
    tvae_sampler: Callable[[pd.DataFrame], pd.DataFrame] = _train_tvae_samples,
    tabddpm_sampler: Callable[[str, pd.DataFrame, str], pd.DataFrame] | None = None,
) -> None:
    _validate_training_input(dataset_name, train_df)

    if model_ids is None:
        model_ids = _benchmark_model_ids()

    if "tabddpm" in model_ids and tabddpm_sampler is None:
        if benchmark_models.get_benchmark_model_spec("tabddpm").trainable:
            tabddpm_sampler = _train_tabddpm_samples
        else:
            raise RuntimeError("TABDDPM benchmark adapter is not configured")

    samplers: dict[str, Callable[[], pd.DataFrame]] = {
        "ctgan": lambda: ctgan_sampler(dataset_name, train_df),
        "tvae": lambda: tvae_sampler(train_df),
        "tabddpm": (
            lambda: tabddpm_sampler(dataset_name, train_df, target_col)
            if tabddpm_sampler is not None
            else (_ for _ in ()).throw(RuntimeError("TABDDPM benchmark adapter is not configured"))
        ),
    }
    saved_outputs: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []

    for model_id in model_ids:
        if model_id not in samplers:
            continue

        synth_df = _coerce_covertype_targets(dataset_name, samplers[model_id](), target_col)
        saved_output, model_warnings = _validate_and_promote_synthetic(
            dataset_name,
            model_id,
            train_df,
            synth_df,
            target_col,
            valid_targets,
        )
        saved_outputs[model_id] = saved_output
        warnings.extend(model_warnings)

    for warning in warnings:
        _warn(dataset_name, warning)

    _print_dataset_completion(dataset_name, target_col, saved_outputs)


def _process_adult_dataset() -> None:
    spec = DATASET_REGISTRY["adult"]
    train_df = pd.read_csv(get_dataset_paths("adult")["train"])
    adult_outputs = copy_adult_synthetic_artifacts()

    warnings: list[str] = []
    for model_id in _benchmark_model_ids():
        saved_output = adult_outputs[model_id]
        warnings.extend(
            _validate_saved_synthetic(
                "adult",
                train_df,
                saved_output,
                spec.target_col,
                spec.valid_target_values,
                expected_rows=None,
            )
        )

    for warning in warnings:
        _warn("adult", warning)

    _print_dataset_completion("adult", spec.target_col, adult_outputs)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ensure_benchmark_dirs()
    model_ids = _benchmark_model_ids()

    dataset_order = ("adult", "bank", "covertype", "diabetes")
    for dataset_name in dataset_order:
        spec = DATASET_REGISTRY[dataset_name]
        if dataset_name == "adult":
            _process_adult_dataset()
            continue

        train_df = pd.read_csv(get_dataset_paths(dataset_name)["train"])
        _process_training_dataset(
            dataset_name,
            train_df=train_df,
            target_col=spec.target_col,
            valid_targets=spec.valid_target_values,
            model_ids=model_ids,
        )

    print("All models trained ✓")


if __name__ == "__main__":
    main()
