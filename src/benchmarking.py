import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.baseline_models import build_baselines
from src.data_loader import DatasetSplit, load_dataset, split_dataset, subsample_training
from src.evaluation import classification_metrics, interval_metrics, metric_directions, regression_metrics
from src.kumo_model import KumoTabularPredictor, KumoUnavailableError, environment_info, resolve_device
from src.visualization import save_classification_figures, save_regression_figures

KUMO_NAME = "Kumo Tabular"


@dataclass
class BenchmarkRun:
    split: DatasetSplit
    results: pd.DataFrame
    predictions: dict[str, np.ndarray] = field(default_factory=dict)
    scores: dict[str, np.ndarray] = field(default_factory=dict)
    kumo_quantiles: pd.DataFrame | None = None


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_kumo_predictor(task: str, kumo_config: dict) -> tuple[KumoTabularPredictor | None, str | None]:
    try:
        predictor = KumoTabularPredictor(
            task=task,
            size=kumo_config["size"],
            device=kumo_config["device"],
            num_estimators=kumo_config["num_estimators"],
            estimator_batch_size=kumo_config["estimator_batch_size"],
            autocast=kumo_config["autocast"],
        )
    except KumoUnavailableError as error:
        return None, str(error)
    return predictor, None


def _base_row(split: DatasetSplit, model_name: str) -> dict:
    return {
        "dataset": split.name,
        "task": split.task,
        "model": model_name,
        "seed": split.random_state,
        "n_train": len(split.X_train),
        "n_test": len(split.X_test),
    }


def _score(split: DatasetSplit, y_pred, y_score=None) -> dict:
    if split.task == "classification":
        return classification_metrics(split.y_test, y_pred, y_score)
    return regression_metrics(split.y_test, y_pred)


def _run_baselines(split: DatasetSplit, config: dict, run: BenchmarkRun) -> list[dict]:
    rows = []
    for name, model in build_baselines(split.task, split.random_state, config.get("baselines")).items():
        start = time.perf_counter()
        model.fit(split.X_train, split.y_train)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        y_pred = model.predict(split.X_test)
        y_score = model.predict_proba(split.X_test)[:, 1] if split.task == "classification" else None
        predict_seconds = time.perf_counter() - start

        run.predictions[name] = np.asarray(y_pred)
        if y_score is not None:
            run.scores[name] = y_score
        rows.append({
            **_base_row(split, name),
            "status": "ok",
            "error": "",
            "device": "cpu",
            **_score(split, y_pred, y_score),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "total_seconds": fit_seconds + predict_seconds,
        })
    return rows


def _run_kumo(
    split: DatasetSplit,
    config: dict,
    run: BenchmarkRun,
    kumo: KumoTabularPredictor | None,
    kumo_error: str | None,
) -> dict:
    row = _base_row(split, KUMO_NAME)
    if kumo is None:
        return {**row, "status": "incomplete", "error": kumo_error or "Kumo Tabular was not loaded", "device": ""}

    kumo_config = config["kumo"]
    try:
        kumo.warmup(split.X_train, split.y_train)
        start = time.perf_counter()
        if split.task == "classification":
            probabilities = kumo.predict_proba(split.X_train, split.y_train, split.X_test, seed=split.random_state)
            predict_seconds = time.perf_counter() - start
            y_score = probabilities[1].to_numpy()
            y_pred = probabilities.idxmax(axis=1).to_numpy().astype(int)
            run.scores[KUMO_NAME] = y_score
            metrics = _score(split, y_pred, y_score)
        else:
            quantiles = kumo.predict_quantiles(split.X_train, split.y_train, split.X_test, seed=split.random_state)
            predict_seconds = time.perf_counter() - start
            y_pred = quantiles[kumo_config["regression_point_quantile"]].to_numpy()
            lower, upper = kumo_config["regression_interval"]
            run.kumo_quantiles = quantiles
            metrics = {
                **_score(split, y_pred),
                **interval_metrics(split.y_test, quantiles[lower].to_numpy(), quantiles[upper].to_numpy()),
            }
    except (RuntimeError, ValueError) as error:
        return {**row, "status": "failed", "error": f"{type(error).__name__}: {error}", "device": str(kumo.device)}

    run.predictions[KUMO_NAME] = y_pred
    return {
        **row,
        "status": "ok",
        "error": "",
        "device": str(kumo.device),
        **metrics,
        "fit_seconds": 0.0,
        "predict_seconds": predict_seconds,
        "total_seconds": predict_seconds,
    }


def run_benchmark(
    split: DatasetSplit,
    config: dict,
    kumo: KumoTabularPredictor | None,
    kumo_error: str | None = None,
) -> BenchmarkRun:
    run = BenchmarkRun(split=split, results=pd.DataFrame())
    rows = [_run_kumo(split, config, run, kumo, kumo_error), *_run_baselines(split, config, run)]
    run.results = pd.DataFrame(rows)
    return run


def run_repeated_benchmark(
    task: str,
    config: dict,
    kumo: KumoTabularPredictor | None,
    kumo_error: str | None = None,
) -> list[BenchmarkRun]:
    name, features, target = load_dataset(task)
    runs = []
    for seed in config["seeds"]:
        split = split_dataset(
            features, target, task=task, name=name, test_size=config["test_size"], random_state=seed
        )
        runs.append(run_benchmark(split, config, kumo, kumo_error))
    return runs


def combine_results(runs: list[BenchmarkRun]) -> pd.DataFrame:
    return pd.concat([run.results for run in runs], ignore_index=True)


def summarize_results(results: pd.DataFrame, task: str) -> pd.DataFrame:
    metrics = list(metric_directions(task))
    if task == "regression" and "interval_coverage" in results:
        metrics += ["interval_coverage", "interval_width"]
    columns = [*metrics, "fit_seconds", "predict_seconds", "total_seconds"]
    completed = results[results["status"] == "ok"]
    order = list(dict.fromkeys(results["model"]))
    summary = completed.groupby("model")[columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reindex(order)
    summary.insert(0, "runs_completed", completed.groupby("model").size().reindex(order).fillna(0).astype(int))
    summary.insert(1, "runs_total", results.groupby("model").size().reindex(order))
    return summary.reset_index()


def metric_leaderboard(summary: pd.DataFrame, task: str) -> pd.DataFrame:
    rows = []
    completed = summary[summary["runs_completed"] > 0]
    baselines = completed[completed["model"] != KUMO_NAME]
    kumo = completed[completed["model"] == KUMO_NAME]
    for metric, higher_is_better in metric_directions(task).items():
        column = f"{metric}_mean"
        pick = pd.Series.idxmax if higher_is_better else pd.Series.idxmin
        best = completed.loc[pick(completed[column])]
        best_baseline = baselines.loc[pick(baselines[column])]
        kumo_value = kumo[column].iloc[0] if len(kumo) else np.nan
        difference = kumo_value - best_baseline[column]
        rows.append({
            "metric": metric,
            "higher_is_better": higher_is_better,
            "best_model": best["model"],
            "best_value": best[column],
            "kumo_value": kumo_value,
            "best_baseline": best_baseline["model"],
            "best_baseline_value": best_baseline[column],
            "kumo_minus_best_baseline": difference,
            "kumo_better": bool(difference > 0 if higher_is_better else difference < 0) if len(kumo) else None,
        })
    return pd.DataFrame(rows)


def paired_comparison(results: pd.DataFrame, leaderboard: pd.DataFrame) -> pd.DataFrame:
    completed = results[results["status"] == "ok"]
    rows = []
    for entry in leaderboard.itertuples():
        by_seed = completed.pivot(index="seed", columns="model", values=entry.metric)
        if KUMO_NAME not in by_seed:
            continue
        paired = by_seed[[KUMO_NAME, entry.best_baseline]].dropna()
        difference = paired[KUMO_NAME] - paired[entry.best_baseline]
        if not entry.higher_is_better:
            difference = -difference
        rows.append({
            "metric": entry.metric,
            "baseline": entry.best_baseline,
            "splits": len(paired),
            "kumo_wins": int((difference > 1e-12).sum()),
            "ties": int((difference.abs() <= 1e-12).sum()),
            "kumo_losses": int((difference < -1e-12).sum()),
            "mean_advantage": difference.mean(),
            "std_advantage": difference.std(),
        })
    return pd.DataFrame(rows)


def run_sample_size_study(
    task: str,
    config: dict,
    kumo: KumoTabularPredictor | None,
    kumo_error: str | None = None,
) -> pd.DataFrame:
    name, features, target = load_dataset(task)
    study = config["sample_size_study"]
    frames = []
    for seed in study["seeds"]:
        split = split_dataset(
            features, target, task=task, name=name, test_size=config["test_size"], random_state=seed
        )
        sizes = [size for size in study["train_sizes"] if size < len(split.X_train)] + [len(split.X_train)]
        for size in sizes:
            subset = subsample_training(split, size, random_state=seed)
            frames.append(run_benchmark(subset, config, kumo, kumo_error).results)
    return pd.concat(frames, ignore_index=True)


def save_table(table: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def save_json(data: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def run_experiment(task: str, config: dict, results_dir: str | Path) -> dict:
    results_dir = Path(results_dir)
    task_dir = results_dir / task
    kumo, kumo_error = create_kumo_predictor(task, config["kumo"])
    if kumo_error:
        print(f"Kumo Tabular unavailable, running baselines only: {kumo_error}")

    runs = run_repeated_benchmark(task, config, kumo, kumo_error)
    results = combine_results(runs)
    summary = summarize_results(results, task)
    leaderboard = metric_leaderboard(summary, task)
    paired = paired_comparison(results, leaderboard)

    save_table(results, task_dir / "results_by_seed.csv")
    save_table(summary, task_dir / "summary.csv")
    save_table(leaderboard, task_dir / "leaderboard.csv")
    save_table(paired, task_dir / "paired_comparison.csv")
    device = kumo.device if kumo else resolve_device("cpu")
    save_json({
        **environment_info(device),
        "kumo_status": "loaded" if kumo else "unavailable",
        "kumo_error": kumo_error,
        "config": config,
    }, task_dir / "environment.json")

    study = run_sample_size_study(task, config, kumo, kumo_error)
    save_table(study, task_dir / "sample_size_study.csv")

    if task == "classification":
        figures = save_classification_figures(runs[0], summary, study, results_dir / "figures")
    else:
        figures = save_regression_figures(runs[0], summary, study, config["kumo"], results_dir / "figures")
    return {"results": results, "summary": summary, "leaderboard": leaderboard, "paired": paired,
            "study": study, "figures": figures}
