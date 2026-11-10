import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.openml_data import load_suite  # noqa: E402

RESULTS = PROJECT_ROOT / "results" / "openml_benchmark"
MODELS = {
    "classification": {"Kumo Tabular", "Logistic Regression", "Random Forest", "Extra Trees", "HistGradientBoosting",
                       "XGBoost", "LightGBM", "CatBoost"},
    "regression": {"Kumo Tabular", "Linear Regression", "Random Forest", "Extra Trees", "HistGradientBoosting",
                   "XGBoost", "LightGBM", "CatBoost"},
}
RANGES = {"accuracy": (0, 1), "macro_f1": (0, 1), "roc_auc": (0, 1), "log_loss": (0, np.inf), "rmse": (0, np.inf),
          "mae": (0, np.inf), "r2": (-np.inf, 1), "fit_seconds": (0, np.inf), "predict_seconds": (0, np.inf),
          "total_seconds": (0, np.inf)}


def check(entry: dict, frame: pd.DataFrame) -> list[str]:
    problems = []
    task = entry["task"]
    metrics = ["accuracy", "macro_f1", "roc_auc", "log_loss"] if task == "classification" else ["rmse", "mae", "r2"]
    if len(frame) != entry["folds"] * len(MODELS[task]):
        problems.append(f"{len(frame)} rows, expected {entry['folds'] * len(MODELS[task])}")
    if set(frame["model"]) != MODELS[task]:
        problems.append(f"models {sorted(set(frame['model']))}")
    if (frame["status"] != "ok").any():
        problems.append(f"non-ok rows: {frame.loc[frame['status'] != 'ok', ['model', 'fold', 'error']].to_dict('records')}")
    for model, group in frame.groupby("model"):
        if sorted(group["fold"]) != list(range(entry["folds"])):
            problems.append(f"{model}: folds {sorted(group['fold'])}")
    sizes = frame.groupby("fold")[["n_train", "n_test"]].first()
    if not (sizes["n_train"] + sizes["n_test"]).eq(entry["rows"]).all():
        problems.append("n_train + n_test differs from the dataset size in some fold")
    if sizes["n_test"].sum() != entry["rows"]:
        problems.append(f"test folds cover {sizes['n_test'].sum()} rows, dataset has {entry['rows']}")
    for column in metrics + ["fit_seconds", "predict_seconds", "total_seconds"]:
        values = frame[column].to_numpy(dtype=float)
        low, high = RANGES[column]
        if not np.isfinite(values).all() or (values < low - 1e-12).any() or (values > high + 1e-12).any():
            problems.append(f"{column} out of range or non-finite")
    if frame[["data_id", "task_id"]].drop_duplicates().to_dict("records") != [
            {"data_id": entry["data_id"], "task_id": entry["task_id"]}]:
        problems.append("data_id/task_id mismatch")
    return problems


def main() -> None:
    suite = load_suite()
    raw_files = sorted((RESULTS / "raw").glob("*.csv"))
    expected = {f"{e['task']}_{e['data_id']}.csv" for e in suite}
    found = {p.name for p in raw_files}
    report, failures = [], 0
    if found != expected:
        print("Unexpected raw files:", sorted(found - expected), "| missing:", sorted(expected - found))
        failures += 1
    for entry in suite:
        frame = pd.read_csv(RESULTS / "raw" / f"{entry['task']}_{entry['data_id']}.csv")
        problems = check(entry, frame)
        failures += bool(problems)
        report.append({"data_id": entry["data_id"], "name": entry["name"], "task": entry["task"], "rows": entry["rows"],
                       "result_rows": len(frame), "ok_rows": int((frame["status"] == "ok").sum()),
                       "problems": problems})
        print(f"{'OK  ' if not problems else 'FAIL'} {entry['task'][:3]} {entry['data_id']:>6} {entry['name']:<34} "
              f"rows={len(frame)} ok={int((frame['status'] == 'ok').sum())} {'; '.join(problems)}")
    combined = pd.read_csv(RESULTS / "results_by_fold.csv")
    concatenated = pd.concat([pd.read_csv(p) for p in raw_files], ignore_index=True)
    same = combined.shape == concatenated.shape and combined.equals(concatenated)
    print(f"results_by_fold.csv equals the concatenated raw files: {same} ({len(combined)} rows)")
    failures += not same
    (RESULTS / "validation_report.json").write_text(json.dumps(
        {"datasets": report, "combined_matches_raw": bool(same), "failures": failures}, indent=2), encoding="utf-8")
    print(f"{len(suite) - sum(bool(r['problems']) for r in report)}/{len(suite)} datasets valid; total failures: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
