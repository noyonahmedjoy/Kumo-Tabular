import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

CLASSIFICATION_METRICS = {"accuracy": True, "precision": True, "recall": True, "f1": True, "roc_auc": True}
REGRESSION_METRICS = {"mae": False, "rmse": False, "r2": True}
SUITE_CLASSIFICATION_METRICS = {"accuracy": True, "macro_f1": True, "roc_auc": True, "log_loss": False}


def metric_directions(task: str) -> dict[str, bool]:
    return CLASSIFICATION_METRICS if task == "classification" else REGRESSION_METRICS


def validate_predictions(y_true, y_pred, name: str = "predictions") -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_pred.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {y_pred.shape}")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"{name} has shape {y_pred.shape}, expected {y_true.shape}")
    if len(y_true) == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(y_pred.astype(float)).all():
        raise ValueError(f"{name} contains non-finite values")
    return y_true, y_pred


def classification_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    y_true, y_pred = validate_predictions(y_true, y_pred, "class predictions")
    _, y_score = validate_predictions(y_true, y_score, "positive-class scores")
    if y_score.min() < 0 or y_score.max() > 1:
        raise ValueError("positive-class scores must be probabilities in [0, 1]")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = validate_predictions(y_true, y_pred, "regression predictions")
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def interval_metrics(y_true, lower, upper) -> dict[str, float]:
    y_true, lower = validate_predictions(y_true, lower, "interval lower bounds")
    _, upper = validate_predictions(y_true, upper, "interval upper bounds")
    if (upper < lower).any():
        raise ValueError("interval upper bounds must not be below lower bounds")
    return {
        "interval_coverage": float(((y_true >= lower) & (y_true <= upper)).mean()),
        "interval_width": float((upper - lower).mean()),
    }


def probabilistic_classification_metrics(y_true, probabilities, n_classes: int) -> dict[str, float]:
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(y_true), n_classes):
        raise ValueError(f"probabilities have shape {probabilities.shape}, expected ({len(y_true)}, {n_classes})")
    if not np.isfinite(probabilities).all() or probabilities.min() < -1e-9:
        raise ValueError("probabilities must be finite and non-negative")
    probabilities = np.clip(probabilities, 1e-15, None)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    y_pred = probabilities.argmax(axis=1)
    labels = list(range(n_classes))
    if n_classes == 2:
        roc_auc = roc_auc_score(y_true, probabilities[:, 1])
    else:
        present = np.unique(y_true)
        scores = probabilities[:, present] / probabilities[:, present].sum(axis=1, keepdims=True)
        roc_auc = roc_auc_score(y_true, scores, multi_class="ovr", average="macro", labels=present) if len(present) > 2 else roc_auc_score(y_true == present[1], scores[:, 1])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "roc_auc": roc_auc,
        "log_loss": log_loss(y_true, probabilities, labels=labels),
    }
