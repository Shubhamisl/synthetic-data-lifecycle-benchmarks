from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from . import benchmark_models
from .common import (
    BENCHMARK_ROOT,
    DATASET_REGISTRY,
    get_dataset_paths,
    validate_dataframe_schema,
    validate_required_non_null_columns,
    validate_target_values,
)

RESULTS_DIR = BENCHMARK_ROOT / "results"
SUMMARY_PATH = RESULTS_DIR / "cross_domain_summary.csv"
MEAN_RANK_PATH = RESULTS_DIR / "mean_rank_table.csv"


def _evaluation_model_specs() -> tuple[benchmark_models.BenchmarkModelSpec, ...]:
    return tuple(
        benchmark_models.get_benchmark_model_spec(model_id)
        for model_id in benchmark_models.get_trainable_benchmark_model_ids()
    )


def demographic_parity_difference(
    df: pd.DataFrame,
    sensitive_col: str,
    target_col: str,
    positive_value: int | str = 1,
) -> float:
    groups = list(df[sensitive_col].dropna().unique())
    if len(groups) != 2:
        raise ValueError("demographic parity requires exactly two groups")

    rates = []
    for group in groups:
        group_df = df[df[sensitive_col] == group]
        rates.append((group_df[target_col] == positive_value).mean())
    return float(abs(rates[0] - rates[1]))


def _kl_divergence(p: np.ndarray, m: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / m)))


def mean_js_divergence(real_df: pd.DataFrame, synth_df: pd.DataFrame, numeric_cols: list[str]) -> float:
    values: list[float] = []
    for column in numeric_cols:
        real = real_df[column].to_numpy(dtype=float)
        synth = synth_df[column].to_numpy(dtype=float)

        lo = min(real.min(), synth.min())
        hi = max(real.max(), synth.max())
        if lo == hi:
            values.append(0.0)
            continue

        bins = np.linspace(lo, hi, 51)
        p = np.histogram(real, bins=bins)[0].astype(float)
        q = np.histogram(synth, bins=bins)[0].astype(float)

        eps = 1e-10
        p = (p / p.sum()) + eps
        q = (q / q.sum()) + eps
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        values.append(0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m))

    return float(np.mean(values)) if values else float("nan")


def rank_models(summary_df: pd.DataFrame) -> pd.DataFrame:
    working = summary_df.copy()
    working["TSTR_Rank"] = working.groupby("Dataset")["TSTR_Accuracy"].rank(ascending=False, method="average")
    working["JS_Rank"] = working.groupby("Dataset")["JS_Divergence"].rank(ascending=True, method="average")
    working["MIA_Rank"] = working.groupby("Dataset")["MIA_Advantage"].rank(ascending=True, method="average")
    working["DP_Rank"] = working.groupby("Dataset")["Demographic_Parity"].rank(ascending=True, method="average")

    ranked = (
        working.groupby("Model", as_index=False)
        .agg(
            Mean_TSTR_Rank=("TSTR_Rank", "mean"),
            Mean_JS_Rank=("JS_Rank", "mean"),
            Mean_MIA_Rank=("MIA_Rank", "mean"),
            Mean_DP_Rank=("DP_Rank", "mean"),
        )
        .sort_values("Model")
        .reset_index(drop=True)
    )
    ranked["Overall_Mean_Rank"] = ranked[
        ["Mean_TSTR_Rank", "Mean_JS_Rank", "Mean_MIA_Rank", "Mean_DP_Rank"]
    ].mean(axis=1, skipna=True)
    return ranked


def _categorical_columns(*dfs: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in dfs[0].columns:
        if any(str(df[column].dtype) in {"object", "category"} for df in dfs if column in df.columns):
            columns.append(column)
    return columns


def _encode_frames(dfs: list[pd.DataFrame], categorical_cols: list[str]) -> list[pd.DataFrame]:
    encoded_frames: list[pd.DataFrame] = []
    mappings: dict[str, dict[str, int]] = {}

    for column in categorical_cols:
        values: list[str] = []
        for df in dfs:
            if column in df.columns:
                values.extend(df[column].astype(str).tolist())
        mappings[column] = {value: index for index, value in enumerate(sorted(set(values)))}

    for df in dfs:
        encoded = df.copy()
        for column in categorical_cols:
            if column in encoded.columns:
                encoded[column] = encoded[column].astype(str).map(mappings[column]).astype(int)
        encoded_frames.append(encoded)
    return encoded_frames


def _numeric_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    return [
        column
        for column in df.select_dtypes(include=[np.number]).columns.tolist()
        if column != target_col
    ]


def tstr_accuracy(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> float:
    categorical_cols = _categorical_columns(train_df, test_df)
    train_enc, test_enc = _encode_frames([train_df, test_df], categorical_cols)

    X_train = train_enc.drop(columns=[target_col])
    y_train = train_enc[target_col]
    X_test = test_enc.drop(columns=[target_col])
    y_test = test_enc[target_col]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test) * 100.0)


def membership_inference_advantage(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
    real = real_df.copy()
    real["_member"] = 1
    synth = synth_df.copy()
    synth["_member"] = 0

    combined = pd.concat([real, synth], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    categorical_cols = _categorical_columns(combined.drop(columns=["_member"]))
    encoded_combined = _encode_frames([combined.drop(columns=["_member"])], categorical_cols)[0]

    X_train, X_test, y_train, y_test = train_test_split(
        encoded_combined,
        combined["_member"],
        test_size=0.2,
        random_state=42,
        stratify=combined["_member"],
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test) - 0.5)


def _validate_evaluation_inputs(
    dataset_name: str,
    real_train: pd.DataFrame,
    real_test: pd.DataFrame,
    synth_df: pd.DataFrame,
    target_col: str,
    sensitive_attr: str | None,
    valid_targets: set[int] | set[str] | None,
) -> None:
    required_columns = real_train.columns.tolist()
    validate_dataframe_schema(real_train, dataset_name, required_columns)
    validate_dataframe_schema(real_test, dataset_name, required_columns)
    validate_dataframe_schema(synth_df, dataset_name, required_columns)
    validate_required_non_null_columns(real_train, dataset_name, required_columns)
    validate_required_non_null_columns(real_test, dataset_name, required_columns)
    validate_required_non_null_columns(synth_df, dataset_name, required_columns)
    validate_target_values(real_train, target_col, valid_targets, dataset_name)
    validate_target_values(real_test, target_col, valid_targets, dataset_name)
    validate_target_values(synth_df, target_col, valid_targets, dataset_name)

    if list(real_train.columns) != list(real_test.columns) or list(real_train.columns) != list(synth_df.columns):
        raise ValueError(f"{dataset_name}: evaluation inputs do not share the same schema")

    if sensitive_attr and sensitive_attr not in synth_df.columns:
        raise ValueError(f"{dataset_name}: missing sensitive attribute {sensitive_attr!r}")


def _display_summary(summary_df: pd.DataFrame) -> str:
    display_df = summary_df.copy()
    display_df["Demographic_Parity"] = display_df["Demographic_Parity"].apply(
        lambda value: "N/A" if pd.isna(value) else f"{value:.4f}"
    )
    for column in ("JS_Divergence", "TSTR_Accuracy", "MIA_Advantage", "TSTR_Real_Baseline"):
        display_df[column] = display_df[column].map(lambda value: f"{value:.4f}")
    return display_df.to_string(index=False)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    model_specs = _evaluation_model_specs()

    for dataset_name in ("adult", "bank", "covertype", "diabetes"):
        spec = DATASET_REGISTRY[dataset_name]
        paths = get_dataset_paths(dataset_name)
        real_train = pd.read_csv(paths["train"])
        real_test = pd.read_csv(paths["test"])
        per_dataset_rows: list[dict[str, object]] = []
        baseline_accuracy = tstr_accuracy(real_train, real_test, spec.target_col)

        for model_spec in model_specs:
            synth_df = pd.read_csv(paths[model_spec.model_id])
            _validate_evaluation_inputs(
                dataset_name,
                real_train,
                real_test,
                synth_df,
                spec.target_col,
                spec.sensitive_attr,
                spec.valid_target_values,
            )

            result_row = {
                "Dataset": dataset_name,
                "Domain": spec.domain,
                "Fidelity_Level": spec.fidelity_level,
                "Model": model_spec.display_name,
                "JS_Divergence": mean_js_divergence(
                    real_train,
                    synth_df,
                    _numeric_feature_columns(real_train, spec.target_col),
                ),
                "TSTR_Accuracy": tstr_accuracy(synth_df, real_test, spec.target_col),
                "MIA_Advantage": membership_inference_advantage(real_train, synth_df),
                "Demographic_Parity": (
                    demographic_parity_difference(synth_df, spec.sensitive_attr, spec.target_col)
                    if spec.sensitive_attr
                    else float("nan")
                ),
                "TSTR_Real_Baseline": baseline_accuracy,
            }
            summary_rows.append(result_row)
            per_dataset_rows.append(
                {
                    "Model": result_row["Model"],
                    "JS_Divergence": result_row["JS_Divergence"],
                    "TSTR_Accuracy": result_row["TSTR_Accuracy"],
                    "MIA_Advantage": result_row["MIA_Advantage"],
                    "Demographic_Parity": result_row["Demographic_Parity"],
                    "TSTR_Real_Baseline": result_row["TSTR_Real_Baseline"],
                }
            )

        pd.DataFrame(per_dataset_rows).to_csv(paths["evaluation"], index=False)

    summary_df = pd.DataFrame(summary_rows)[
        [
            "Dataset",
            "Domain",
            "Fidelity_Level",
            "Model",
            "JS_Divergence",
            "TSTR_Accuracy",
            "MIA_Advantage",
            "Demographic_Parity",
            "TSTR_Real_Baseline",
        ]
    ]
    summary_df.to_csv(SUMMARY_PATH, index=False)

    rank_df = rank_models(summary_df)[
        [
            "Model",
            "Mean_TSTR_Rank",
            "Mean_JS_Rank",
            "Mean_MIA_Rank",
            "Mean_DP_Rank",
            "Overall_Mean_Rank",
        ]
    ]
    rank_df.to_csv(MEAN_RANK_PATH, index=False)

    print(_display_summary(summary_df))


if __name__ == "__main__":
    main()
