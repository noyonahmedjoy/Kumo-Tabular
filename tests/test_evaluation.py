import numpy as np
import pytest

from src.evaluation import classification_metrics, interval_metrics, regression_metrics, validate_predictions


def test_classification_metrics_match_hand_computation():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.4, 0.6, 0.3, 0.2, 0.1, 0.05])
    metrics = classification_metrics(y_true, y_pred, y_score)
    assert metrics["accuracy"] == pytest.approx(6 / 8)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["roc_auc"] == pytest.approx(14 / 15)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (4, 1, 1, 2)


def test_regression_metrics_match_hand_computation():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 3.0, 2.0, 6.0])
    metrics = regression_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(1.5))
    assert metrics["r2"] == pytest.approx(1 - 6 / 5)


def test_perfect_regression():
    y = np.array([3.0, 1.0, 2.0])
    assert regression_metrics(y, y) == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}


@pytest.mark.parametrize("y_pred", [np.zeros(3), np.zeros((4, 2)), np.zeros(5)])
def test_invalid_prediction_shapes_are_rejected(y_pred):
    with pytest.raises(ValueError):
        validate_predictions(np.zeros(4), y_pred)


def test_non_finite_predictions_are_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        regression_metrics(np.ones(3), np.array([1.0, np.nan, 2.0]))


def test_empty_predictions_are_rejected():
    with pytest.raises(ValueError, match="empty"):
        regression_metrics(np.array([]), np.array([]))


def test_class_score_outside_probability_range_is_rejected():
    with pytest.raises(ValueError, match="probabilities"):
        classification_metrics(np.array([0, 1]), np.array([0, 1]), np.array([-0.2, 1.3]))


def test_interval_metrics():
    metrics = interval_metrics(np.array([1.0, 5.0, 9.0]), np.array([0.0, 6.0, 8.0]), np.array([2.0, 7.0, 10.0]))
    assert metrics["interval_coverage"] == pytest.approx(2 / 3)
    assert metrics["interval_width"] == pytest.approx(5 / 3)


def test_inverted_interval_is_rejected():
    with pytest.raises(ValueError):
        interval_metrics(np.ones(2), np.array([2.0, 2.0]), np.array([1.0, 3.0]))
