import io
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = PROJECT_ROOT / "configs" / "openml_suite.json"
CACHE_DIR = PROJECT_ROOT / "data" / "openml_cache"


@dataclass(frozen=True)
class OpenMLTask:
    data_id: int
    task_id: int
    name: str
    task: str
    X: pd.DataFrame
    y: pd.Series
    classes: list
    folds: list[tuple[np.ndarray, np.ndarray]]


def load_suite(path: Path = SUITE_PATH) -> list[dict]:
    suite = json.loads(path.read_text(encoding="utf-8"))
    return [{**entry, "task": task} for task in ("classification", "regression") for entry in suite[task]]


def parse_splits(arff_text: str, repeat: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    body = arff_text.split("@DATA", 1)[-1] if "@DATA" in arff_text else arff_text.split("@data", 1)[-1]
    frame = pd.read_csv(io.StringIO(body), header=None, names=["type", "rowid", "repeat", "fold"], comment="%")
    frame["type"] = frame["type"].str.strip().str.strip("'\"").str.upper()
    frame = frame[frame["repeat"] == repeat]
    folds = []
    for fold in sorted(frame["fold"].unique()):
        part = frame[frame["fold"] == fold]
        train = np.sort(part.loc[part["type"] == "TRAIN", "rowid"].to_numpy(dtype=int))
        test = np.sort(part.loc[part["type"] == "TEST", "rowid"].to_numpy(dtype=int))
        if len(np.intersect1d(train, test)) > 0:
            raise ValueError(f"fold {fold} has rows in both train and test")
        folds.append((train, test))
    return folds


def task_splits(task_id: int, splits_url: str, cache_dir: Path = CACHE_DIR) -> list[tuple[np.ndarray, np.ndarray]]:
    path = cache_dir / "splits" / f"task_{task_id}_splits.arff"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(splits_url, timeout=300) as response:
            path.write_bytes(response.read())
    return parse_splits(path.read_text(encoding="utf-8", errors="replace"))


def load_openml_task(entry: dict, cache_dir: Path = CACHE_DIR) -> OpenMLTask:
    bunch = fetch_openml(
        data_id=entry["data_id"], target_column=entry["target"], as_frame=True, parser="liac-arff",
        data_home=str(cache_dir))
    X = bunch.data.apply(pd.to_numeric, errors="raise").astype(float).reset_index(drop=True)
    X.columns = [str(column) for column in X.columns]
    target = bunch.target.reset_index(drop=True)
    if entry["task"] == "classification":
        classes = sorted(target.astype(str).unique())
        y = target.astype(str).map({label: index for index, label in enumerate(classes)}).astype(int)
    else:
        classes = []
        y = pd.to_numeric(target, errors="raise").astype(float)
    if X.isna().any().any() or y.isna().any():
        raise ValueError(f"{entry['name']} contains missing values")
    folds = task_splits(entry["task_id"], entry["splits_url"], cache_dir)
    covered = np.sort(np.concatenate([test for _, test in folds]))
    if not np.array_equal(covered, np.arange(len(X))):
        raise ValueError(f"{entry['name']}: test folds do not cover every row exactly once")
    return OpenMLTask(entry["data_id"], entry["task_id"], entry["name"], entry["task"], X, y.rename("target"),
                      classes, folds)
