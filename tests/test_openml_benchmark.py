import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation import probabilistic_classification_metrics
from src.kumo_model import QUANTILE_COLUMNS
from src.openml_benchmark import KUMO, _full_probabilities, run_task
from src.openml_data import OpenMLTask, parse_splits

ARFF = """@RELATION splits
@ATTRIBUTE type {TRAIN,TEST}
@ATTRIBUTE rowid NUMERIC
@ATTRIBUTE repeat NUMERIC
@ATTRIBUTE fold NUMERIC
@DATA
TRAIN,2,0,0
TRAIN,3,0,0
TEST,0,0,0
TEST,1,0,0
TRAIN,0,0,1
TRAIN,1,0,1
TEST,2,0,1
TEST,3,0,1
TRAIN,0,1,0
TEST,3,1,0
"""


def test_parse_splits_reads_first_repeat():
    folds = parse_splits(ARFF)
    assert len(folds) == 2
    assert folds[0][0].tolist() == [2, 3] and folds[0][1].tolist() == [0, 1]
    assert folds[1][0].tolist() == [0, 1] and folds[1][1].tolist() == [2, 3]


def test_parse_splits_rejects_overlap():
    with pytest.raises(ValueError, match="both train and test"):
        parse_splits(ARFF.replace("TRAIN,3,0,0", "TRAIN,0,0,0"))


def test_probabilistic_metrics_match_hand_computation():
    metrics = probabilistic_classification_metrics([0, 1, 1, 0], [[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.3, 0.7]], 2)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["log_loss"] == pytest.approx(-np.mean(np.log([0.9, 0.8, 0.6, 0.3])))


def test_probabilistic_metrics_multiclass_and_shape_check():
    probabilities = [[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7], [0.3, 0.3, 0.4]]
    metrics = probabilistic_classification_metrics([0, 1, 2, 2], probabilities, 3)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="shape"):
        probabilistic_classification_metrics([0, 1], probabilities, 3)


def test_full_probabilities_fills_missing_classes():
    frame = pd.DataFrame({0: [0.6, 0.2], 2: [0.4, 0.8]})
    full = _full_probabilities(frame, 3)
    assert full.tolist() == [[0.6, 0.0, 0.4], [0.2, 0.0, 0.8]]


class StubKumo:
    def __init__(self):
        self.device = torch.device("cpu")
        self.is_warm = False
        self.seen = []

    def warmup(self, X_context, y_context):
        self.is_warm = True

    def predict_proba(self, X_context, y_context, X_query, seed=0):
        assert X_context.index.intersection(X_query.index).empty
        self.seen.append(set(y_context.index))
        classes = np.sort(y_context.unique())
        return pd.DataFrame(np.full((len(X_query), len(classes)), 1 / len(classes)), columns=classes,
                            index=X_query.index)

    def predict_quantiles(self, X_context, y_context, X_query, seed=0):
        assert X_context.index.intersection(X_query.index).empty
        self.seen.append(set(y_context.index))
        values = np.quantile(y_context.to_numpy(), np.linspace(0.001, 0.999, 999))
        return pd.DataFrame(np.tile(values, (len(X_query), 1)), columns=QUANTILE_COLUMNS, index=X_query.index)


def synthetic_task(task: str) -> OpenMLTask:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    y = pd.Series((X["a"] > 0).astype(int) if task == "classification" else X["a"] * 2 + rng.normal(size=60))
    folds = [(np.setdiff1d(np.arange(60), np.arange(i, 60, 3)), np.arange(i, 60, 3)) for i in range(3)]
    return OpenMLTask(1, 2, "synthetic", task, X, y, [0, 1] if task == "classification" else [], folds)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_run_task_covers_all_models_without_leakage(task):
    stub = StubKumo()
    data = synthetic_task(task)
    results = run_task(data, {task: stub})
    assert (results["status"] == "ok").all()
    assert results.groupby("model").size().eq(3).all()
    assert results["model"].nunique() == 8
    for (_, test_index), seen in zip(data.folds, stub.seen):
        assert seen.isdisjoint(set(test_index))


def test_run_task_reports_missing_kumo():
    results = run_task(synthetic_task("classification"), {})
    kumo_rows = results[results["model"] == KUMO]
    assert (kumo_rows["status"] == "incomplete").all()
    assert (results[results["model"] != KUMO]["status"] == "ok").all()
