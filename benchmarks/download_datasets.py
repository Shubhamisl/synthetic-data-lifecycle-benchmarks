from __future__ import annotations

import shutil
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from .common import (
    BENCHMARK_ROOT,
    DATASET_REGISTRY,
    PROJECT_ROOT,
    ensure_benchmark_dirs,
    format_summary_table,
    get_dataset_paths,
    log_benchmark_event,
    validate_dataframe_schema,
    validate_required_non_null_columns,
    validate_target_values,
)

BANK_DIRECT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
COVERTYPE_DIRECT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
DIABETES_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
ADULT_EVALUATION_SOURCE = PROJECT_ROOT / "results" / "final_evaluation_table.csv"
ADULT_TRAIN_SOURCE = PROJECT_ROOT / "data" / "adult_train.csv"
ADULT_TEST_SOURCE = PROJECT_ROOT / "data" / "adult_test.csv"

DIABETES_COLUMNS = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
    "target",
]

DIABETES_ZERO_MISSING_COLUMNS = [
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
]
COVERTYPE_COLUMNS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
    *[f"Wilderness_Area{i}" for i in range(1, 5)],
    *[f"Soil_Type{i}" for i in range(1, 41)],
    "Cover_Type",
]
ADULT_EVALUATION_REQUIRED_COLUMNS = [
    "Model",
    "JS_Divergence",
    "TSTR_Accuracy",
    "MIA_Advantage",
    "Demographic_Parity",
]
ADULT_EVALUATION_REQUIRED_MODELS = {"Real", "CTGAN", "TVAE"}
ADULT_EVALUATION_SYNTHETIC_MODELS = {"CTGAN", "TVAE"}
ADULT_EVALUATION_REQUIRED_SYNTHETIC_METRICS = [
    "JS_Divergence",
    "TSTR_Accuracy",
    "MIA_Advantage",
    "Demographic_Parity",
]


def split_stratified(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def preprocess_bank(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    observed_labels = set(out["y"].dropna().unique().tolist())
    valid_labels = {"yes", "no"}
    invalid_labels = observed_labels - valid_labels
    if invalid_labels:
        raise ValueError(f"bank: unexpected target labels {sorted(invalid_labels)}")
    out = out.rename(columns={"y": "target"})
    categorical_columns = out.select_dtypes(include=["object", "category"]).columns.tolist()
    for column in categorical_columns:
        if column != "target":
            out[column] = out[column].fillna("unknown")
    out["target"] = out["target"].map({"yes": 1, "no": 0})
    out["age_group"] = out["age"].apply(lambda value: "young" if value <= 40 else "older")
    return out.drop(columns=[column for column in ("age", "duration") if column in out.columns])


def preprocess_diabetes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in DIABETES_ZERO_MISSING_COLUMNS:
        out[column] = out[column].astype(float)
        median_value = out.loc[out[column] != 0, column].median()
        if pd.isna(median_value):
            raise ValueError(f"diabetes: unable to compute non-zero median for {column}")
        out.loc[out[column] == 0, column] = median_value
    out["age_group"] = out["age"].apply(lambda value: "young" if value <= 35 else "older")
    return out.drop(columns=["age"])


def summarize_dataset(
    name: str,
    df: pd.DataFrame,
    target_col: str,
    sensitive_attr: str | None,
) -> dict[str, object]:
    return {
        "Dataset": name,
        "Samples": len(df),
        "Features": len(df.columns),
        "Target classes": df[target_col].nunique(dropna=True),
        "Sensitive attr": sensitive_attr or "None",
    }


def validate_saved_dataset_pair(
    dataset_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    sensitive_attr: str | None,
    valid_target_values: set[int] | set[str] | None,
) -> None:
    validate_dataframe_schema(train_df, dataset_name, required_columns=[target_col])
    validate_dataframe_schema(test_df, dataset_name, required_columns=[target_col])
    validate_required_non_null_columns(train_df, dataset_name, [target_col])
    validate_required_non_null_columns(test_df, dataset_name, [target_col])

    if sensitive_attr:
        if sensitive_attr not in train_df.columns:
            raise ValueError(f"{dataset_name}: missing sensitive attribute {sensitive_attr!r} in train split")
        if sensitive_attr not in test_df.columns:
            raise ValueError(f"{dataset_name}: missing sensitive attribute {sensitive_attr!r} in test split")
        validate_required_non_null_columns(train_df, dataset_name, [sensitive_attr])
        validate_required_non_null_columns(test_df, dataset_name, [sensitive_attr])

    if list(train_df.columns) != list(test_df.columns):
        raise ValueError(f"{dataset_name}: train/test columns do not match exactly")

    validate_target_values(train_df, target_col, valid_target_values, dataset_name)
    validate_target_values(test_df, target_col, valid_target_values, dataset_name)

    train_classes = set(train_df[target_col].dropna().unique().tolist())
    test_classes = set(test_df[target_col].dropna().unique().tolist())
    if not train_classes or not test_classes:
        raise ValueError(f"{dataset_name}: stratified split produced an empty target class set")
    if train_classes != test_classes:
        raise ValueError(f"{dataset_name}: stratification was not preserved across train/test splits")

    if dataset_name == "diabetes":
        zero_total = (train_df[DIABETES_ZERO_MISSING_COLUMNS] == 0).sum().sum()
        zero_total += (test_df[DIABETES_ZERO_MISSING_COLUMNS] == 0).sum().sum()
        if zero_total:
            raise ValueError("diabetes: zero-missing replacements did not succeed")

    if dataset_name == "covertype":
        valid_set = valid_target_values or set()
        observed = train_classes | test_classes
        if not observed.issubset(valid_set):
            raise ValueError(f"{dataset_name}: invalid class set detected {sorted(observed - valid_set)}")


def load_bank_dataframe_with_fallback() -> pd.DataFrame:
    try:
        dataset = fetch_ucirepo(id=222)
        return pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    except Exception as exc:
        log_benchmark_event("download", "bank", "WARNING", f"ucimlrepo failed, using direct UCI fallback: {exc}")

    try:
        with urlopen(BANK_DIRECT_URL) as response:
            payload = response.read()
        with ZipFile(BytesIO(payload)) as archive:
            with archive.open("bank-full.csv") as handle:
                return pd.read_csv(handle, sep=";")
    except Exception as exc:
        raise RuntimeError(f"bank: unable to fetch dataset via ucimlrepo or direct UCI fallback: {exc}") from exc


def load_covertype_dataframe_with_fallback() -> pd.DataFrame:
    try:
        dataset = fetch_ucirepo(id=31)
        frame = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        frame["Cover_Type"] = pd.to_numeric(frame["Cover_Type"], errors="raise").astype(int)
        return frame
    except Exception as exc:
        log_benchmark_event("download", "covertype", "WARNING", f"ucimlrepo failed, using direct UCI fallback: {exc}")

    try:
        frame = pd.read_csv(
            COVERTYPE_DIRECT_URL,
            header=None,
            names=COVERTYPE_COLUMNS,
            compression="gzip",
        )
        frame["Cover_Type"] = pd.to_numeric(frame["Cover_Type"], errors="raise").astype(int)
        return frame
    except Exception as exc:
        raise RuntimeError(f"covertype: unable to fetch dataset via ucimlrepo or direct UCI fallback: {exc}") from exc


def load_diabetes_dataframe() -> pd.DataFrame:
    try:
        return pd.read_csv(DIABETES_URL, header=None, names=DIABETES_COLUMNS)
    except Exception as exc:
        raise RuntimeError(f"diabetes: unable to fetch dataset from GitHub CSV: {exc}") from exc


def stratified_sample(
    df: pd.DataFrame,
    target_col: str,
    sample_fraction: float = 0.10,
    fallback_fraction: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    validate_dataframe_schema(df, "covertype", [target_col])
    validate_required_non_null_columns(df, "covertype", [target_col])

    try:
        sample_df, _ = train_test_split(
            df,
            train_size=sample_fraction,
            stratify=df[target_col],
            random_state=random_state,
        )
        return sample_df.reset_index(drop=True)
    except ValueError as exc:
        if not _is_legitimate_stratified_sample_error(str(exc)):
            raise
        log_benchmark_event(
            "download",
            "covertype",
            "WARNING",
            f"10% stratified sample failed ({exc}); retrying with 5%",
        )
        sample_df, _ = train_test_split(
            df,
            train_size=fallback_fraction,
            stratify=df[target_col],
            random_state=random_state,
        )
        return sample_df.reset_index(drop=True)


def _is_legitimate_stratified_sample_error(message: str) -> bool:
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in (
            "least populated class in y",
            "train_size =",
            "test_size =",
            "should be greater or equal to the number of classes",
        )
    )


def print_dataset_details(dataset_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> None:
    combined = pd.concat([train_df, test_df], ignore_index=True)
    column_info = ", ".join(f"{column}:{dtype}" for column, dtype in combined.dtypes.items())
    print(f"[{dataset_name}]")
    print(f"  Train shape: {train_df.shape}")
    print(f"  Test shape: {test_df.shape}")
    print(f"  Train class balance: {train_df[target_col].value_counts(normalize=True).sort_index().round(4).to_dict()}")
    print(f"  Test class balance: {test_df[target_col].value_counts(normalize=True).sort_index().round(4).to_dict()}")
    print(f"  Columns: {column_info}")


def copy_adult_artifacts() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_benchmark_dirs()
    paths = get_dataset_paths("adult")
    shutil.copy2(ADULT_TRAIN_SOURCE, paths["train"])
    shutil.copy2(ADULT_TEST_SOURCE, paths["test"])
    shutil.copy2(ADULT_EVALUATION_SOURCE, paths["evaluation"])
    return pd.read_csv(paths["train"]), pd.read_csv(paths["test"])


def validate_adult_evaluation_csv(path) -> pd.DataFrame:
    csv_path = path
    if not csv_path.exists():
        raise ValueError(f"adult evaluation: missing copied evaluation CSV at {csv_path}")

    evaluation_df = pd.read_csv(csv_path)
    validate_dataframe_schema(evaluation_df, "adult evaluation", ADULT_EVALUATION_REQUIRED_COLUMNS)
    validate_required_non_null_columns(evaluation_df, "adult evaluation", ["Model", "TSTR_Accuracy"])

    observed_models = set(evaluation_df["Model"].unique().tolist())
    missing_models = ADULT_EVALUATION_REQUIRED_MODELS - observed_models
    if missing_models:
        raise ValueError(f"adult evaluation: missing required model rows {sorted(missing_models)}")

    synthetic_rows = evaluation_df[evaluation_df["Model"].isin(ADULT_EVALUATION_SYNTHETIC_MODELS)]
    for metric in ADULT_EVALUATION_REQUIRED_SYNTHETIC_METRICS:
        if synthetic_rows[metric].isna().any():
            raise ValueError(f"adult evaluation: null values in required synthetic metric {metric!r}")

    return evaluation_df


def load_and_validate_adult_dataset_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = DATASET_REGISTRY["adult"]
    train_df, test_df = copy_adult_artifacts()
    validate_adult_evaluation_csv(get_dataset_paths("adult")["evaluation"])
    validate_saved_dataset_pair(
        "adult",
        train_df,
        test_df,
        target_col=spec.target_col,
        sensitive_attr=spec.sensitive_attr,
        valid_target_values=spec.valid_target_values,
    )
    return train_df, test_df


def save_dataset_pair(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = get_dataset_paths(name)
    train_df.to_csv(paths["train"], index=False)
    test_df.to_csv(paths["test"], index=False)
    return pd.read_csv(paths["train"]), pd.read_csv(paths["test"])


def process_dataset(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = DATASET_REGISTRY[name]
    if name == "adult":
        return load_and_validate_adult_dataset_pair()
    if name == "bank":
        processed = preprocess_bank(load_bank_dataframe_with_fallback())
        validate_dataframe_schema(processed, name, [spec.target_col, spec.sensitive_attr or spec.target_col])
        train_df, test_df = split_stratified(processed, spec.target_col)
        return save_dataset_pair(name, train_df, test_df)
    if name == "covertype":
        raw_df = load_covertype_dataframe_with_fallback()
        validate_dataframe_schema(raw_df, name, [spec.target_col])
        validate_target_values(raw_df, spec.target_col, spec.valid_target_values, name)
        sampled_df = stratified_sample(raw_df, spec.target_col)
        train_df, test_df = split_stratified(sampled_df, spec.target_col)
        return save_dataset_pair(name, train_df, test_df)
    if name == "diabetes":
        processed = preprocess_diabetes(load_diabetes_dataframe())
        validate_dataframe_schema(processed, name, [spec.target_col, spec.sensitive_attr or spec.target_col])
        train_df, test_df = split_stratified(processed, spec.target_col)
        return save_dataset_pair(name, train_df, test_df)
    raise ValueError(f"Unsupported dataset {name!r}")


def main() -> None:
    ensure_benchmark_dirs()
    summary_rows: list[dict[str, object]] = []

    for dataset_name in ("adult", "bank", "covertype", "diabetes"):
        spec = DATASET_REGISTRY[dataset_name]
        try:
            train_df, test_df = process_dataset(dataset_name)
            validate_saved_dataset_pair(
                dataset_name,
                train_df,
                test_df,
                target_col=spec.target_col,
                sensitive_attr=spec.sensitive_attr,
                valid_target_values=spec.valid_target_values,
            )
            print_dataset_details(dataset_name, train_df, test_df, spec.target_col)
            combined_df = pd.concat([train_df, test_df], ignore_index=True)
            summary_rows.append(
                summarize_dataset(
                    dataset_name,
                    combined_df,
                    target_col=spec.target_col,
                    sensitive_attr=spec.sensitive_attr,
                )
            )
        except Exception as exc:
            message = f"Failed to prepare {dataset_name} dataset: {exc}"
            log_benchmark_event("download", dataset_name, "ERROR", message)
            print(message)
            raise RuntimeError(message) from exc

    print("Final summary")
    print(format_summary_table(summary_rows))


if __name__ == "__main__":
    main()
