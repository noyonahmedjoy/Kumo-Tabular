from dataclasses import dataclass, replace

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "target"


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    task: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    random_state: int


def load_classification_data() -> tuple[pd.DataFrame, pd.Series]:
    bunch = load_breast_cancer(as_frame=True)
    malignant_code = list(bunch.target_names).index("malignant")
    is_malignant = (bunch.target == malignant_code).astype(int)
    return bunch.data, is_malignant.rename(TARGET_COLUMN)


def load_regression_data() -> tuple[pd.DataFrame, pd.Series]:
    bunch = load_diabetes(as_frame=True, scaled=False)
    return bunch.data, bunch.target.astype(float).rename(TARGET_COLUMN)


DATASETS = {
    "classification": ("breast_cancer", load_classification_data),
    "regression": ("diabetes", load_regression_data),
}


def load_dataset(task: str) -> tuple[str, pd.DataFrame, pd.Series]:
    if task not in DATASETS:
        raise ValueError(f"Unknown task '{task}', expected one of {sorted(DATASETS)}")
    name, loader = DATASETS[task]
    features, target = loader()
    return name, features, target


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    task: str,
    name: str,
    test_size: float,
    random_state: int,
) -> DatasetSplit:
    if TARGET_COLUMN in features.columns:
        raise ValueError(f"Feature frame must not contain the '{TARGET_COLUMN}' column")
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target if task == "classification" else None,
    )
    split = DatasetSplit(name, task, X_train, X_test, y_train, y_test, random_state)
    validate_split(split)
    return split


def validate_split(split: DatasetSplit) -> None:
    overlap = split.X_train.index.intersection(split.X_test.index)
    if len(overlap) > 0:
        raise ValueError(f"{len(overlap)} rows appear in both the training and test sets")
    if not split.X_train.index.equals(split.y_train.index):
        raise ValueError("Training features and targets are not aligned")
    if not split.X_test.index.equals(split.y_test.index):
        raise ValueError("Test features and targets are not aligned")
    if list(split.X_train.columns) != list(split.X_test.columns):
        raise ValueError("Training and test feature columns differ")
    if TARGET_COLUMN in split.X_train.columns or TARGET_COLUMN in split.X_test.columns:
        raise ValueError(f"The '{TARGET_COLUMN}' column leaked into the feature frames")


def subsample_training(split: DatasetSplit, n_rows: int, random_state: int) -> DatasetSplit:
    if n_rows >= len(split.X_train):
        return split
    X_train, _, y_train, _ = train_test_split(
        split.X_train,
        split.y_train,
        train_size=n_rows,
        random_state=random_state,
        stratify=split.y_train if split.task == "classification" else None,
    )
    subset = replace(split, X_train=X_train, y_train=y_train)
    validate_split(subset)
    return subset
