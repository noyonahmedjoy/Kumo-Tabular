import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.benchmarking import create_kumo_predictor, load_config  # noqa: E402
from src.baseline_models import suite_baselines  # noqa: E402
from src.kumo_model import environment_info, resolve_device  # noqa: E402
from src.openml_benchmark import completed, run_task  # noqa: E402
from src.openml_data import load_openml_task, load_suite  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "results" / "openml_benchmark"
PACKAGES = ["torch", "structured-data-models", "scikit-learn", "xgboost", "lightgbm", "catboost", "numpy", "pandas"]


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with open(OUTPUT_DIR / "run_log.txt", "a", encoding="utf-8") as file:
        file.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OpenML small-data benchmark.")
    parser.add_argument("--only", type=int, nargs="*", help="OpenML data ids to run")
    args = parser.parse_args()

    (OUTPUT_DIR / "raw").mkdir(parents=True, exist_ok=True)
    config = load_config(PROJECT_ROOT / "configs" / "experiment_config.yaml")
    kumo_by_task, errors = {}, {}
    for task in ("classification", "regression"):
        kumo_by_task[task], errors[task] = create_kumo_predictor(task, config["kumo"])
        log(f"Kumo {task}: {'loaded' if kumo_by_task[task] else errors[task]}")

    device = next((k.device for k in kumo_by_task.values() if k is not None), resolve_device("cpu"))
    environment = {
        **environment_info(device),
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "processor": platform.processor(),
        "kumo": config["kumo"],
        "kumo_errors": errors,
    }
    (OUTPUT_DIR / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    for entry in load_suite():
        if args.only and entry["data_id"] not in args.only:
            continue
        path = OUTPUT_DIR / "raw" / f"{entry['task']}_{entry['data_id']}.csv"
        n_models = len(suite_baselines(entry["task"], 0)) + 1
        if completed(path, entry["folds"], n_models):
            log(f"skip {entry['name']} (already complete)")
            continue
        start = time.perf_counter()
        task = load_openml_task(entry)
        results = run_task(task, kumo_by_task, config["kumo"]["regression_point_quantile"])
        results.to_csv(path, index=False)
        failed = results[results["status"] != "ok"]
        log(f"done {entry['name']} ({entry['task']}, {len(task.X)} rows, {len(task.folds)} folds) "
            f"in {time.perf_counter() - start:.0f}s; failed rows: {len(failed)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    frames = [pd.read_csv(p) for p in sorted((OUTPUT_DIR / "raw").glob("*.csv"))]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(OUTPUT_DIR / "results_by_fold.csv", index=False)
        log(f"combined {len(frames)} datasets into results_by_fold.csv")


if __name__ == "__main__":
    main()
