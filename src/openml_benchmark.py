import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.baseline_models import suite_baselines
from src.evaluation import probabilistic_classification_metrics, regression_metrics
from src.kumo_model import KumoTabularPredictor
from src.openml_data import OpenMLTask

KUMO = "Kumo Tabular"


def _full_probabilities(probabilities: pd.DataFrame, n_classes: int) -> np.ndarray:
    full = np.zeros((len(probabilities), n_classes))
    for label in probabilities.columns:
        full[:, int(label)] = probabilities[label].to_numpy()
    return full


def _baseline_probabilities(model, X_test: pd.DataFrame, n_classes: int) -> np.ndarray:
    raw = model.predict_proba(X_test)
    classes = np.asarray(model.classes_ if hasattr(model, "classes_") else model[-1].classes_).astype(int)
    full = np.zeros((len(X_test), n_classes))
    full[:, classes] = raw
    return full


def _metrics(task: OpenMLTask, y_test: pd.Series, output) -> dict:
    if task.task == "classification":
        return probabilistic_classification_metrics(y_test.to_numpy(), output, len(task.classes))
    return regression_metrics(y_test.to_numpy(), output)


def _row(task: OpenMLTask, fold: int, model: str, n_train: int, n_test: int) -> dict:
    return {"data_id": task.data_id, "task_id": task.task_id, "dataset": task.name, "task": task.task,
            "fold": fold, "model": model, "n_train": n_train, "n_test": n_test}


def run_task(task: OpenMLTask, kumo_by_task: dict, kumo_point_quantile: str = "q500") -> pd.DataFrame:
    rows = []
    kumo = kumo_by_task.get(task.task)
    for fold, (train_index, test_index) in enumerate(task.folds):
        X_train, X_test = task.X.iloc[train_index], task.X.iloc[test_index]
        y_train, y_test = task.y.iloc[train_index], task.y.iloc[test_index]
        base = lambda model: _row(task, fold, model, len(train_index), len(test_index))

        if kumo is None:
            rows.append({**base(KUMO), "status": "incomplete", "error": "Kumo Tabular not loaded"})
        else:
            try:
                if fold == 0:
                    kumo.is_warm = False
                    kumo.warmup(X_train, y_train)
                start = time.perf_counter()
                if task.task == "classification":
                    probabilities = kumo.predict_proba(X_train, y_train, X_test, seed=fold)
                    output = _full_probabilities(probabilities, len(task.classes))
                else:
                    output = kumo.predict_quantiles(X_train, y_train, X_test, seed=fold)[kumo_point_quantile].to_numpy()
                if kumo.device.type == "cuda":
                    torch.cuda.synchronize(kumo.device)
                seconds = time.perf_counter() - start
                rows.append({**base(KUMO), "status": "ok", "error": "", "device": str(kumo.device),
                             **_metrics(task, y_test, output), "fit_seconds": 0.0, "predict_seconds": seconds,
                             "total_seconds": seconds})
            except (RuntimeError, ValueError) as error:
                rows.append({**base(KUMO), "status": "failed", "error": f"{type(error).__name__}: {error}"})

        for name, model in suite_baselines(task.task, random_state=fold).items():
            try:
                start = time.perf_counter()
                model.fit(X_train, y_train)
                fit_seconds = time.perf_counter() - start
                start = time.perf_counter()
                if task.task == "classification":
                    output = _baseline_probabilities(model, X_test, len(task.classes))
                else:
                    output = np.asarray(model.predict(X_test), dtype=float)
                predict_seconds = time.perf_counter() - start
                rows.append({**base(name), "status": "ok", "error": "", "device": "cpu",
                             **_metrics(task, y_test, output), "fit_seconds": fit_seconds,
                             "predict_seconds": predict_seconds, "total_seconds": fit_seconds + predict_seconds})
            except (RuntimeError, ValueError) as error:
                rows.append({**base(name), "status": "failed", "error": f"{type(error).__name__}: {error}"})
    return pd.DataFrame(rows)


def completed(path: Path, n_folds: int, n_models: int) -> bool:
    if not path.exists():
        return False
    frame = pd.read_csv(path)
    return len(frame) == n_folds * n_models
