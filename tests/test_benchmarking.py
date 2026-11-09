import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from src.benchmarking import (  # noqa: E402
    KUMO_NAME,
    combine_results,
    create_kumo_predictor,
    load_config,
    metric_leaderboard,
    paired_comparison,
    run_benchmark,
    run_repeated_benchmark,
    run_sample_size_study,
    save_table,
    summarize_results,
)
from src.data_loader import TARGET_COLUMN, load_dataset, split_dataset  # noqa: E402
from src.kumo_model import QUANTILE_COLUMNS, KumoTabularPredictor, KumoUnavailableError  # noqa: E402
from src.visualization import save_classification_figures, save_regression_figures  # noqa: E402

CONFIG = load_config("configs/experiment_config.yaml")
CONFIG["seeds"] = [42, 43]
CONFIG["sample_size_study"] = {"train_sizes": [40, 120], "seeds": [42]}
CONFIG["baselines"]["random_forest"]["n_estimators"] = 50


class RecordingKumo:
    def __init__(self, task, output_shape_error=False):
        self.task = task
        self.device = torch.device("cpu")
        self.output_shape_error = output_shape_error
        self.seen_targets = []

    def warmup(self, X_context, y_context):
        pass

    def _check(self, X_context, y_context, X_query):
        assert TARGET_COLUMN not in X_query.columns
        assert X_context.index.intersection(X_query.index).empty
        self.seen_targets.append(set(y_context.index))

    def predict_proba(self, X_context, y_context, X_query, seed=0):
        self._check(X_context, y_context, X_query)
        if self.output_shape_error:
            raise ValueError("Kumo returned shape (3, 2), expected (114, 2)")
        rate = y_context.mean()
        return pd.DataFrame({0: np.full(len(X_query), 1 - rate), 1: np.full(len(X_query), rate)},
                            index=X_query.index)

    def predict_quantiles(self, X_context, y_context, X_query, seed=0):
        self._check(X_context, y_context, X_query)
        levels = np.linspace(0.001, 0.999, 999)
        values = np.quantile(y_context.to_numpy(), levels)
        return pd.DataFrame(np.tile(values, (len(X_query), 1)), columns=QUANTILE_COLUMNS, index=X_query.index)


def make_split(task, seed=42):
    name, features, target = load_dataset(task)
    return split_dataset(features, target, task=task, name=name, test_size=0.2, random_state=seed)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_benchmark_runs_all_models(task):
    kumo = RecordingKumo(task)
    split = make_split(task)
    run = run_benchmark(split, CONFIG, kumo)
    assert list(run.results["model"])[0] == KUMO_NAME
    assert len(run.results) == 4
    assert (run.results["status"] == "ok").all()
    assert all(len(prediction) == len(split.X_test) for prediction in run.predictions.values())


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_kumo_never_sees_test_targets(task):
    kumo = RecordingKumo(task)
    split = make_split(task)
    run_benchmark(split, CONFIG, kumo)
    test_rows = set(split.X_test.index)
    assert kumo.seen_targets and all(seen.isdisjoint(test_rows) for seen in kumo.seen_targets)


def test_missing_kumo_is_reported_as_incomplete():
    run = run_benchmark(make_split("classification"), CONFIG, None, "weights unavailable")
    kumo_row = run.results.iloc[0]
    assert kumo_row["status"] == "incomplete"
    assert kumo_row["error"] == "weights unavailable"
    assert np.isnan(kumo_row["accuracy"])
    assert KUMO_NAME not in run.predictions
    assert (run.results.iloc[1:]["status"] == "ok").all()


def test_invalid_kumo_output_is_reported_as_failed():
    run = run_benchmark(make_split("classification"), CONFIG, RecordingKumo("classification", True))
    kumo_row = run.results.iloc[0]
    assert kumo_row["status"] == "failed"
    assert "expected (114, 2)" in kumo_row["error"]
    assert np.isnan(kumo_row["accuracy"])


def test_summary_and_leaderboard_exclude_incomplete_runs():
    runs = run_repeated_benchmark("classification", CONFIG, None, "not installed")
    summary = summarize_results(combine_results(runs), "classification")
    kumo_summary = summary[summary["model"] == KUMO_NAME].iloc[0]
    assert kumo_summary["runs_completed"] == 0
    assert kumo_summary["runs_total"] == 2
    leaderboard = metric_leaderboard(summary, "classification")
    assert (leaderboard["best_model"] != KUMO_NAME).all()
    assert leaderboard["kumo_value"].isna().all()


def test_leaderboard_respects_metric_direction():
    runs = run_repeated_benchmark("regression", CONFIG, RecordingKumo("regression"))
    summary = summarize_results(combine_results(runs), "regression")
    leaderboard = metric_leaderboard(summary, "regression").set_index("metric")
    assert leaderboard.loc["mae", "best_value"] == summary["mae_mean"].min()
    assert leaderboard.loc["r2", "best_value"] == summary["r2_mean"].max()
    assert summary["runs_completed"].eq(2).all()


def test_paired_comparison_counts_every_split():
    runs = run_repeated_benchmark("regression", CONFIG, RecordingKumo("regression"))
    results = combine_results(runs)
    leaderboard = metric_leaderboard(summarize_results(results, "regression"), "regression")
    paired = paired_comparison(results, leaderboard)
    assert list(paired["metric"]) == ["mae", "rmse", "r2"]
    assert (paired["kumo_wins"] + paired["ties"] + paired["kumo_losses"]).eq(2).all()


def test_paired_comparison_skips_missing_kumo():
    runs = run_repeated_benchmark("classification", CONFIG, None, "not installed")
    results = combine_results(runs)
    leaderboard = metric_leaderboard(summarize_results(results, "classification"), "classification")
    assert paired_comparison(results, leaderboard).empty


def test_sample_size_study_uses_requested_sizes():
    study = run_sample_size_study("regression", CONFIG, RecordingKumo("regression"))
    assert sorted(study["n_train"].unique()) == [40, 120, 353]
    assert study.groupby("n_train")["n_test"].nunique().eq(1).all()


def test_results_export(tmp_path):
    runs = run_repeated_benchmark("classification", CONFIG, RecordingKumo("classification"))
    results = combine_results(runs)
    path = save_table(results, tmp_path / "nested" / "results.csv")
    loaded = pd.read_csv(path)
    assert loaded.shape == results.shape
    assert list(loaded["model"]) == list(results["model"])


def test_figures_are_written(tmp_path):
    kumo = RecordingKumo("classification")
    classification_runs = run_repeated_benchmark("classification", CONFIG, kumo)
    summary = summarize_results(combine_results(classification_runs), "classification")
    study = run_sample_size_study("classification", CONFIG, kumo)
    paths = save_classification_figures(classification_runs[0], summary, study, tmp_path)

    kumo = RecordingKumo("regression")
    regression_runs = run_repeated_benchmark("regression", CONFIG, kumo)
    summary = summarize_results(combine_results(regression_runs), "regression")
    study = run_sample_size_study("regression", CONFIG, kumo)
    paths += save_regression_figures(regression_runs[0], summary, study, CONFIG["kumo"], tmp_path)

    assert len(paths) == 12
    assert all(path.exists() and path.stat().st_size > 10_000 for path in paths)


def test_unusable_device_is_reported_instead_of_raised():
    kumo_config = {**CONFIG["kumo"], "device": "cuda:99"}
    kumo, error = create_kumo_predictor("classification", kumo_config)
    assert kumo is None
    assert "cuda" in error.lower()


def test_kumo_rejects_target_column_in_query():
    try:
        kumo = KumoTabularPredictor("classification", size="small", device="cpu", num_estimators=1)
    except KumoUnavailableError as error:
        pytest.skip(f"Kumo Tabular unavailable: {error}")
    split = make_split("classification")
    with pytest.raises(ValueError, match="must not contain"):
        kumo.predict_proba(split.X_train, split.y_train, split.X_test.assign(target=split.y_test))


def test_kumo_inference_shapes():
    try:
        kumo = KumoTabularPredictor("classification", size="small", device="auto", num_estimators=2)
    except KumoUnavailableError as error:
        pytest.skip(f"Kumo Tabular unavailable: {error}")
    split = make_split("classification")
    probabilities = kumo.predict_proba(split.X_train, split.y_train, split.X_test, seed=0)
    assert probabilities.shape == (len(split.X_test), 2)
    assert list(probabilities.columns) == [0, 1]
    assert probabilities.index.equals(split.X_test.index)
    assert ((probabilities[1] > 0.5).astype(int) == split.y_test).mean() > 0.9
