import pandas as pd
import pytest

from src.data_loader import (
    TARGET_COLUMN,
    load_classification_data,
    load_dataset,
    load_regression_data,
    split_dataset,
    subsample_training,
    validate_split,
)


@pytest.fixture(scope="module")
def classification_split():
    name, features, target = load_dataset("classification")
    return split_dataset(features, target, task="classification", name=name, test_size=0.2, random_state=42)


def test_classification_data_shape_and_labels():
    features, target = load_classification_data()
    assert features.shape == (569, 30)
    assert set(target.unique()) == {0, 1}
    assert target.sum() == 212
    assert TARGET_COLUMN not in features.columns


def test_regression_data_is_unscaled():
    features, target = load_regression_data()
    assert features.shape == (442, 10)
    assert target.dtype == float
    assert features["age"].between(18, 90).all()


def test_unknown_task_raises():
    with pytest.raises(ValueError):
        load_dataset("ranking")


def test_split_is_disjoint_and_complete(classification_split):
    split = classification_split
    assert split.X_train.index.intersection(split.X_test.index).empty
    assert len(split.X_train) + len(split.X_test) == 569
    assert len(split.X_test) == 114


def test_split_is_stratified(classification_split):
    split = classification_split
    assert abs(split.y_train.mean() - split.y_test.mean()) < 0.02


def test_split_is_reproducible(classification_split):
    name, features, target = load_dataset("classification")
    again = split_dataset(features, target, task="classification", name=name, test_size=0.2, random_state=42)
    assert again.X_test.index.equals(classification_split.X_test.index)


def test_target_column_in_features_is_rejected():
    features, target = load_classification_data()
    with pytest.raises(ValueError, match="must not contain"):
        split_dataset(features.assign(target=target), target, task="classification", name="leak",
                      test_size=0.2, random_state=0)


def test_validate_split_detects_overlap(classification_split):
    split = classification_split
    leaked_rows = split.X_test.iloc[:5]
    leaked = type(split)(
        split.name, split.task,
        pd.concat([split.X_train, leaked_rows]), split.X_test,
        pd.concat([split.y_train, split.y_test.iloc[:5]]), split.y_test,
        split.random_state,
    )
    with pytest.raises(ValueError, match="both the training and test"):
        validate_split(leaked)


def test_subsample_keeps_test_set_and_classes(classification_split):
    subset = subsample_training(classification_split, 50, random_state=0)
    assert len(subset.X_train) == 50
    assert subset.X_test.index.equals(classification_split.X_test.index)
    assert set(subset.y_train.unique()) == {0, 1}
    assert subset.X_train.index.isin(classification_split.X_train.index).all()


def test_subsample_larger_than_train_returns_full_split(classification_split):
    assert subsample_training(classification_split, 10_000, random_state=0) is classification_split
