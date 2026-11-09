import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
SOURCE_MODULES = [
    "data_loader.py",
    "kumo_model.py",
    "baseline_models.py",
    "evaluation.py",
    "visualization.py",
    "benchmarking.py",
]

ENVIRONMENT_NOTES = """## Environment

The notebook runs on Kaggle, Colab or a local Jupyter installation.

On Kaggle, open **Settings** in the notebook editor and check two things before running:

- **Internet** must be switched on. It is needed once to install `structured-data-models` and to download the
  Kumo Tabular `small` checkpoint (about 110 MB per task) from Hugging Face. No account or token is required.
- **Accelerator** is optional. A GPU (T4) makes Kumo inference much faster; on CPU the notebook still completes,
  only more slowly. The traditional baselines always run on CPU.

If the package cannot be installed or the weights cannot be downloaded, the notebook prints the error,
marks the Kumo Tabular rows as `incomplete` and continues with the baseline models. Baseline-only results are
never labelled as Kumo results."""

INSTALL_CELL = """import importlib.util
import subprocess
import sys

if importlib.util.find_spec("sdm") is None:
    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "structured-data-models==0.2.1"],
        capture_output=True,
        text=True,
    )
    if install.returncode == 0:
        print("Installed structured-data-models 0.2.1")
    else:
        print("Could not install structured-data-models. Kumo Tabular will be skipped.")
        print(install.stderr[-1500:])
else:
    print("structured-data-models is already installed")"""

ENVIRONMENT_CELL = """import os
import platform
import socket

import torch

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def can_reach(host, port=443, timeout=5):
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


print(f"Python          {platform.python_version()}")
print(f"PyTorch         {torch.__version__} (CUDA build {torch.version.cuda})")
print(f"CUDA available  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    properties = torch.cuda.get_device_properties(0)
    print(f"GPU             {properties.name}, {properties.total_memory / 1024**3:.1f} GB")
else:
    print("GPU             none, Kumo Tabular will run on CPU")
print(f"Hugging Face    {'reachable' if can_reach('huggingface.co') else 'NOT reachable, enable Internet to download weights'}")"""

WORKSPACE_CELL = """import tempfile
from pathlib import Path

WORK_DIR = Path(tempfile.gettempdir()) / "tabular_intelligence"
SRC_PATH = (WORK_DIR / "src").as_posix()
CONFIG_PATH = (WORK_DIR / "experiment_config.yaml").as_posix()
OUTPUT_DIR = Path("outputs")
(WORK_DIR / "src").mkdir(parents=True, exist_ok=True)
(WORK_DIR / "src" / "__init__.py").touch()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(WORK_DIR))
print("Project code is written to a temporary directory; CSV files and figures are saved to ./outputs")"""

SOURCE_NOTES = """## Project code

The cells below write the project modules from the Tabular Intelligence repository into a temporary
directory so that the notebook is self-contained. They are generated from `src/` by
`scripts/build_notebooks.py` and are identical to the repository code.

- `data_loader.py` loads the datasets and builds leakage-checked train/test splits.
- `kumo_model.py` wraps the official `sdm.models.KumoTabular` API.
- `baseline_models.py` defines the scikit-learn baselines.
- `evaluation.py` computes and validates metrics.
- `visualization.py` produces the figures.
- `benchmarking.py` runs every model on identical splits and aggregates the results."""


def source_cells() -> list:
    cells = [new_markdown_cell(SOURCE_NOTES)]
    for module in SOURCE_MODULES:
        code = (PROJECT_ROOT / "src" / module).read_text(encoding="utf-8").rstrip()
        cells.append(new_code_cell(f"%%writefile {{SRC_PATH}}/{module}\n{code}"))
    config = (PROJECT_ROOT / "configs" / "experiment_config.yaml").read_text(encoding="utf-8").rstrip()
    cells.append(new_code_cell(f"%%writefile {{CONFIG_PATH}}\n{config}"))
    return cells


def setup_cells() -> list:
    return [
        new_markdown_cell(ENVIRONMENT_NOTES),
        new_code_cell(INSTALL_CELL),
        new_code_cell(ENVIRONMENT_CELL),
        new_code_cell(WORKSPACE_CELL),
        *source_cells(),
    ]


IMPORT_CELL = """import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from src.benchmarking import (
    KUMO_NAME,
    combine_results,
    create_kumo_predictor,
    load_config,
    metric_leaderboard,
    paired_comparison,
    run_repeated_benchmark,
    run_sample_size_study,
    save_json,
    save_table,
    summarize_results,
)
from src.data_loader import load_dataset
from src.kumo_model import environment_info, resolve_device
from src import visualization as viz

warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*", category=UserWarning)
pd.set_option("display.precision", 4)
pd.set_option("display.width", 160)
config = load_config(CONFIG_PATH)
config"""


def classification_notebook() -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell("""# Tabular Intelligence: Breast Cancer Classification

Benchmarking NVIDIA Kumo Tabular against traditional machine learning models.

This notebook compares the pretrained [Kumo Tabular](https://huggingface.co/nvidia/Kumo-Tabular) foundation model
with Logistic Regression, Random Forest and HistGradientBoosting on the Breast Cancer Wisconsin Diagnostic dataset.
Kumo Tabular is not trained on this dataset. It receives the labelled training rows as context and predicts the
test rows in a single forward pass (in-context learning)."""),
        new_markdown_cell("""## Methodology

- **Data.** 569 tumours, 30 numeric features computed from digitised images of fine needle aspirates.
  The label is recoded so that `1 = malignant` (212 rows) and `0 = benign` (357 rows). Malignant is the positive
  class for precision, recall and F1, because missing a malignant tumour is the costly error.
- **Splits.** Stratified 80/20 train/test splits, repeated over five seeds (42 to 46). Every model sees exactly the
  same rows for a given seed.
- **Kumo Tabular.** The training rows and their labels are the context; the test rows are passed as query rows
  without the target column. The wrapper rejects query frames that contain the target and context rows that overlap
  with the query rows. Model size `small`, 8 ensemble estimators, seeded generator.
- **Baselines.** scikit-learn defaults with a fixed `random_state`; Logistic Regression uses standardised inputs and
  Random Forest uses 500 trees. No model is tuned, so the comparison is between out-of-the-box methods.
- **Metrics.** Accuracy, precision, recall, F1, ROC-AUC and the confusion matrix, reported as mean and standard
  deviation over the five splits. Runtime is fit plus predict time per split; Kumo has no fit step, so all of its
  work is counted as prediction time. A warm-up call on training rows is excluded from timing."""),
        *setup_cells(),
        new_markdown_cell("## Data"),
        new_code_cell(IMPORT_CELL),
        new_code_cell("""dataset_name, features, target = load_dataset("classification")
print(f"{dataset_name}: {features.shape[0]} rows, {features.shape[1]} features")
print(target.map({0: "benign", 1: "malignant"}).value_counts().to_string())
features.describe().T[["mean", "std", "min", "max"]].head(10)"""),
        new_markdown_cell("""## Load Kumo Tabular

The model is loaded once and reused for every split. If loading fails, the error is shown and the benchmark
continues with the baselines only."""),
        new_code_cell("""kumo, kumo_error = create_kumo_predictor("classification", config["kumo"])
if kumo is None:
    print(f"Kumo Tabular unavailable: {kumo_error}")
else:
    print(f"Loaded Kumo Tabular ({kumo.size}) on {kumo.device}, autocast={'on' if kumo.autocast else 'off'}")
environment = environment_info(kumo.device if kumo else resolve_device("cpu"))
environment"""),
        new_markdown_cell("## Run the benchmark"),
        new_code_cell("""runs = run_repeated_benchmark("classification", config, kumo, kumo_error)
results = combine_results(runs)
summary = summarize_results(results, "classification")
leaderboard = metric_leaderboard(summary, "classification")
paired = paired_comparison(results, leaderboard)

save_table(results, OUTPUT_DIR / "classification_results_by_seed.csv")
save_table(summary, OUTPUT_DIR / "classification_summary.csv")
save_table(leaderboard, OUTPUT_DIR / "classification_leaderboard.csv")
save_table(paired, OUTPUT_DIR / "classification_paired_comparison.csv")
save_json({**environment, "kumo_error": kumo_error}, OUTPUT_DIR / "classification_environment.json")

results[["model", "seed", "status", "accuracy", "precision", "recall", "f1", "roc_auc", "total_seconds"]]"""),
        new_markdown_cell("""## Results

Mean and standard deviation over the five splits. `runs_completed` shows how many splits produced a valid result;
a Kumo row with zero completed runs means the model could not execute."""),
        new_code_cell("""metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc", "total_seconds"]
summary[["model", "runs_completed"] + [f"{m}_{s}" for m in metric_columns for s in ("mean", "std")]]"""),
        new_markdown_cell("""Best model per metric, and the paired comparison between Kumo Tabular and the strongest baseline for each
metric on the same splits. A win means Kumo was better on that split."""),
        new_code_cell("""display(leaderboard)
display(paired)"""),
        new_markdown_cell("## Visualisations"),
        new_code_cell("""figure = viz.plot_metric_comparison(summary, ["accuracy", "precision", "recall", "f1", "roc_auc"],
                                   "Breast cancer classification: mean and standard deviation across splits")
viz.save_figure(figure, OUTPUT_DIR / "classification_metrics.png")
plt.show()"""),
        new_markdown_cell("The following plots use the first split (seed 42)."),
        new_code_cell("""primary = runs[0]
figure = viz.plot_confusion_matrices(primary.split.y_test, primary.predictions, ["benign", "malignant"])
viz.save_figure(figure, OUTPUT_DIR / "classification_confusion_matrices.png")
plt.show()"""),
        new_code_cell("""figure = viz.plot_roc_curves(primary.split.y_test, primary.scores)
viz.save_figure(figure, OUTPUT_DIR / "classification_roc_curves.png")
plt.show()
figure = viz.plot_precision_recall_curves(primary.split.y_test, primary.scores)
viz.save_figure(figure, OUTPUT_DIR / "classification_precision_recall.png")
plt.show()"""),
        new_code_cell("""figure = viz.plot_inference_time(summary, "Classification runtime")
viz.save_figure(figure, OUTPUT_DIR / "classification_inference_time.png")
plt.show()"""),
        new_markdown_cell("""## Error analysis

Which test rows were misclassified on each split, and by which models. Rows that every model gets wrong usually
point to ambiguous samples rather than a weakness of one method."""),
        new_code_cell("""error_rows = []
for run in runs:
    for row_id in run.split.X_test.index:
        position = run.split.X_test.index.get_loc(row_id)
        wrong = [model for model, prediction in run.predictions.items()
                 if prediction[position] != run.split.y_test.loc[row_id]]
        if wrong:
            error_rows.append({
                "seed": run.split.random_state,
                "row": row_id,
                "actual": "malignant" if run.split.y_test.loc[row_id] == 1 else "benign",
                "kumo_p_malignant": run.scores[KUMO_NAME][position] if KUMO_NAME in run.scores else np.nan,
                "misclassified_by": ", ".join(wrong),
                "models_wrong": len(wrong),
            })
errors = pd.DataFrame(error_rows)
save_table(errors, OUTPUT_DIR / "classification_errors.csv")
errors.sort_values(["models_wrong", "seed"], ascending=[False, True]).head(20)"""),
        new_code_cell("""error_counts = results[results["status"] == "ok"].groupby("model")[["fp", "fn"]].sum()
error_counts.columns = ["false positives (benign called malignant)", "false negatives (missed malignant)"]
error_counts"""),
        new_markdown_cell("""## Conclusions

The statements below describe the run stored with this notebook (Kumo Tabular `small`, five stratified splits).
Re-running on other hardware can change runtimes, but the predictions are seeded.

- Kumo Tabular had the highest mean on all five metrics, with accuracy 0.990 against 0.977 for Logistic Regression,
  the strongest baseline. Its accuracy was higher than Logistic Regression on all five splits.
- The largest gain was in recall on the malignant class (0.976 against 0.948), i.e. fewer missed malignant tumours.
  Over the five test sets Kumo produced 5 false negatives, against 11 for Logistic Regression.
- ROC-AUC is close to saturated for every model (0.988 to 0.998), so it separates the methods less than recall does.
- The standard deviations overlap. With 114 test rows per split, one misclassified row changes accuracy by 0.009,
  so the differences are consistent but small in absolute terms.
- Kumo needed about 1.1 s per split on a GTX 980M, compared with 0.02 s for Logistic Regression and about 1.3 s for
  a 500-tree Random Forest. On a CPU-only machine Kumo is roughly ten to twenty times slower than on the GPU."""),
    ]
    return new_notebook(cells=cells, metadata=kernel_metadata())


def regression_notebook() -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell("""# Tabular Intelligence: Diabetes Progression Regression

Benchmarking NVIDIA Kumo Tabular against traditional machine learning models.

This notebook compares the pretrained [Kumo Tabular](https://huggingface.co/nvidia/Kumo-Tabular) regressor with
Linear Regression, Random Forest and HistGradientBoosting on the scikit-learn diabetes dataset. Kumo Tabular
predicts 999 quantiles of the target for every query row, which also allows a check of its prediction intervals."""),
        new_markdown_cell("""## Methodology

- **Data.** 442 patients, 10 baseline variables (age, sex, BMI, blood pressure and six serum measurements) in their
  original units. The target is a quantitative measure of disease progression one year after baseline.
- **Splits.** Random 80/20 train/test splits over five seeds (42 to 46), shared by all models.
- **Kumo Tabular.** Training rows and targets are the context; test rows without the target are the query.
  The model returns quantiles `q001` to `q999`. The **median, `q500`, is the point prediction** used for MAE, RMSE
  and R². The interval from `q050` to `q950` is a nominal 90% prediction interval and is evaluated for coverage.
- **Baselines.** scikit-learn defaults with a fixed `random_state`, Random Forest with 500 trees, no tuning.
- **Metrics.** MAE, RMSE and R² as mean and standard deviation over the splits, plus fit and predict time."""),
        *setup_cells(),
        new_markdown_cell("## Data"),
        new_code_cell(IMPORT_CELL),
        new_code_cell("""dataset_name, features, target = load_dataset("regression")
print(f"{dataset_name}: {features.shape[0]} rows, {features.shape[1]} features")
print(target.describe().round(1).to_string())
features.head()"""),
        new_markdown_cell("## Load Kumo Tabular"),
        new_code_cell("""kumo, kumo_error = create_kumo_predictor("regression", config["kumo"])
if kumo is None:
    print(f"Kumo Tabular unavailable: {kumo_error}")
else:
    print(f"Loaded Kumo Tabular ({kumo.size}) on {kumo.device}, autocast={'on' if kumo.autocast else 'off'}")
environment = environment_info(kumo.device if kumo else resolve_device("cpu"))
environment"""),
        new_markdown_cell("## Run the benchmark"),
        new_code_cell("""runs = run_repeated_benchmark("regression", config, kumo, kumo_error)
results = combine_results(runs)
summary = summarize_results(results, "regression")
leaderboard = metric_leaderboard(summary, "regression")
paired = paired_comparison(results, leaderboard)

save_table(results, OUTPUT_DIR / "regression_results_by_seed.csv")
save_table(summary, OUTPUT_DIR / "regression_summary.csv")
save_table(leaderboard, OUTPUT_DIR / "regression_leaderboard.csv")
save_table(paired, OUTPUT_DIR / "regression_paired_comparison.csv")
save_json({**environment, "kumo_error": kumo_error}, OUTPUT_DIR / "regression_environment.json")

results[["model", "seed", "status", "mae", "rmse", "r2", "total_seconds"]]"""),
        new_markdown_cell("## Results"),
        new_code_cell("""metric_columns = ["mae", "rmse", "r2", "total_seconds"]
summary[["model", "runs_completed"] + [f"{m}_{s}" for m in metric_columns for s in ("mean", "std")]]"""),
        new_code_cell("""display(leaderboard)
display(paired)"""),
        new_markdown_cell("## Visualisations"),
        new_code_cell("""figure = viz.plot_metric_comparison(summary, ["mae", "rmse", "r2"],
                                   "Diabetes regression: mean and standard deviation across splits")
viz.save_figure(figure, OUTPUT_DIR / "regression_metrics.png")
plt.show()"""),
        new_markdown_cell("The following plots use the first split (seed 42)."),
        new_code_cell("""primary = runs[0]
figure = viz.plot_actual_vs_predicted(primary.split.y_test, primary.predictions)
viz.save_figure(figure, OUTPUT_DIR / "regression_actual_vs_predicted.png")
plt.show()
figure = viz.plot_residual_distributions(primary.split.y_test, primary.predictions)
viz.save_figure(figure, OUTPUT_DIR / "regression_residuals.png")
plt.show()"""),
        new_code_cell("""figure = viz.plot_inference_time(summary, "Regression runtime")
viz.save_figure(figure, OUTPUT_DIR / "regression_inference_time.png")
plt.show()"""),
        new_markdown_cell("""## Quantile outputs

Kumo Tabular returns a predictive distribution rather than a single number. The plot shows the `q050` to `q950`
interval for every test row of the first split. For a well calibrated model about 90% of the actual values fall
inside it."""),
        new_code_cell("""if primary.kumo_quantiles is not None:
    lower, upper = config["kumo"]["regression_interval"]
    figure = viz.plot_prediction_intervals(primary.split.y_test, primary.kumo_quantiles,
                                           config["kumo"]["regression_point_quantile"], lower, upper)
    viz.save_figure(figure, OUTPUT_DIR / "regression_kumo_intervals.png")
    plt.show()
    display(summary.loc[summary["model"] == KUMO_NAME,
                        ["interval_coverage_mean", "interval_coverage_std", "interval_width_mean"]])
else:
    print("Kumo Tabular did not run, no quantile outputs to show.")"""),
        new_markdown_cell("""The choice of point estimate matters for squared-error metrics. The median minimises absolute error, while the
mean of the predictive distribution minimises squared error. The comparison below uses the stored quantiles of
each split, so no additional model calls are made. The averaged quantile grid is used as an approximation of the
predictive mean."""),
        new_code_cell("""from src.evaluation import regression_metrics

point_rows = []
for run in runs:
    if run.kumo_quantiles is None:
        continue
    for label, values in [("median (q500)", run.kumo_quantiles["q500"]),
                          ("mean of quantiles", run.kumo_quantiles.mean(axis=1))]:
        point_rows.append({"seed": run.split.random_state, "point_estimate": label,
                           **regression_metrics(run.split.y_test, values.to_numpy())})
point_estimates = pd.DataFrame(point_rows)
point_estimates.groupby("point_estimate")[["mae", "rmse", "r2"]].mean() if len(point_estimates) else point_estimates"""),
        new_markdown_cell("""## Error analysis

The test rows with the largest absolute Kumo errors on the first split, with the corresponding baseline errors."""),
        new_code_cell("""if KUMO_NAME in primary.predictions:
    errors = pd.DataFrame({"actual": primary.split.y_test})
    for model, prediction in primary.predictions.items():
        errors[f"{model} error"] = prediction - primary.split.y_test.to_numpy()
    errors = errors.join(primary.split.X_test[["age", "bmi", "bp", "s5"]])
    errors["kumo_abs_error"] = errors[f"{KUMO_NAME} error"].abs()
    save_table(errors.reset_index(names="row"), OUTPUT_DIR / "regression_errors.csv")
    display(errors.sort_values("kumo_abs_error", ascending=False).head(10).round(1))
    print("Correlation of absolute errors with Kumo:")
    print(errors.filter(like=" error").abs().corr()[f"{KUMO_NAME} error"].round(2).to_string())"""),
        new_markdown_cell("""## Conclusions

The statements below describe the run stored with this notebook (Kumo Tabular `small`, five random splits).

- Kumo Tabular had the best mean MAE (42.5), RMSE (52.9) and R² (0.503). Linear Regression was the strongest
  baseline with MAE 43.7, RMSE 53.3 and R² 0.496. The tree ensembles were clearly behind (R² 0.44 and 0.39).
- The advantage over Linear Regression is not consistent: Kumo had the lower MAE on four of the five splits but the
  better R² on only three, and the mean R² difference (0.007) is far smaller than the split-to-split standard
  deviation (about 0.04). On this dataset the two methods should be considered equivalent.
- The point estimate matters: averaging Kumo's quantiles instead of taking the median lowered RMSE from 52.9 to
  52.4 and raised R² from 0.503 to 0.513, while the median kept the better MAE. This matches the loss each
  estimate minimises. The median remains the reported point prediction.
- The diabetes target is noisy and largely linear in the features, which is why a simple linear model is hard
  to beat here. Large errors are shared across models, so they reflect the data rather than one method.
- The nominal 90% interval from `q050` to `q950` covered 93% of the test targets on average, which is slightly
  conservative but reasonably calibrated. None of the baselines provides intervals without additional work.
- Kumo needed about 0.6 s per split on a GTX 980M, against 0.003 s for Linear Regression."""),
    ]
    return new_notebook(cells=cells, metadata=kernel_metadata())


def analysis_notebook() -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell("""# Tabular Intelligence: Results Analysis

Benchmarking NVIDIA Kumo Tabular against traditional machine learning models.

This notebook reruns both benchmarks from scratch, brings the results together and adds a training-size study.
It does not read any files produced by the other notebooks."""),
        new_markdown_cell("""## Questions

1. On which metrics does Kumo Tabular beat the traditional baselines, and is the difference consistent across
   splits?
2. What does in-context learning cost in runtime compared with training a classical model?
3. How does performance change with the number of labelled rows? Foundation models are often claimed to be
   most useful when labelled data is scarce; the training-size study measures this directly by giving every model
   the same subsample of the training set (25 to all rows) and evaluating on the full test set."""),
        *setup_cells(),
        new_markdown_cell("## Run both benchmarks"),
        new_code_cell(IMPORT_CELL),
        new_code_cell("""experiments = {}
for task in ["classification", "regression"]:
    kumo, kumo_error = create_kumo_predictor(task, config["kumo"])
    if kumo is None:
        print(f"{task}: Kumo Tabular unavailable: {kumo_error}")
    runs = run_repeated_benchmark(task, config, kumo, kumo_error)
    results = combine_results(runs)
    summary = summarize_results(results, task)
    leaderboard = metric_leaderboard(summary, task)
    experiments[task] = {
        "kumo": kumo,
        "kumo_error": kumo_error,
        "results": results,
        "summary": summary,
        "leaderboard": leaderboard,
        "paired": paired_comparison(results, leaderboard),
    }
    save_table(results, OUTPUT_DIR / f"{task}_results_by_seed.csv")
    save_table(summary, OUTPUT_DIR / f"{task}_summary.csv")
    print(f"{task}: {len(results)} model runs, {int((results['status'] == 'ok').sum())} completed")

device = next((e["kumo"].device for e in experiments.values() if e["kumo"] is not None), resolve_device("cpu"))
environment_info(device)"""),
        new_markdown_cell("## Performance summary"),
        new_code_cell("""def format_summary(summary, metrics):
    table = summary[["model"]].copy()
    for metric in metrics:
        table[metric] = [f"{mean:.3f} ± {std:.3f}" if pd.notna(mean) else "incomplete"
                         for mean, std in zip(summary[f"{metric}_mean"], summary[f"{metric}_std"])]
    table["seconds per split"] = summary["total_seconds_mean"].map(lambda value: f"{value:.3f}")
    return table


display(format_summary(experiments["classification"]["summary"], ["accuracy", "precision", "recall", "f1", "roc_auc"]))
display(format_summary(experiments["regression"]["summary"], ["mae", "rmse", "r2"]))"""),
        new_markdown_cell("## Best model per metric"),
        new_code_cell("""leaderboards = pd.concat(
    [experiments[task]["leaderboard"].assign(task=task) for task in experiments], ignore_index=True
)
save_table(leaderboards, OUTPUT_DIR / "leaderboard.csv")
leaderboards[["task", "metric", "best_model", "best_value", "kumo_value", "best_baseline",
              "best_baseline_value", "kumo_minus_best_baseline", "kumo_better"]]"""),
        new_markdown_cell("""## Consistency across splits

A higher mean can come from one lucky split. The paired comparison counts on how many of the five splits Kumo
Tabular beat the strongest baseline for each metric, and reports the mean and standard deviation of its advantage
(positive means Kumo is better, also for error metrics)."""),
        new_code_cell("""paired = pd.concat([experiments[task]["paired"].assign(task=task) for task in experiments], ignore_index=True)
save_table(paired, OUTPUT_DIR / "paired_comparison.csv")
paired"""),
        new_markdown_cell("## Runtime"),
        new_code_cell("""for task, title in [("classification", "Classification runtime"), ("regression", "Regression runtime")]:
    figure = viz.plot_inference_time(experiments[task]["summary"], title)
    viz.save_figure(figure, OUTPUT_DIR / f"{task}_inference_time.png")
    plt.show()"""),
        new_markdown_cell("""## Training-size study

Each model is trained (or, for Kumo, conditioned) on a stratified subsample of the training split and evaluated
on the untouched test split. Three seeds per size; the error bars show the standard deviation."""),
        new_code_cell("""studies = {}
for task in experiments:
    studies[task] = run_sample_size_study(task, config, experiments[task]["kumo"], experiments[task]["kumo_error"])
    save_table(studies[task], OUTPUT_DIR / f"{task}_sample_size_study.csv")

for task, metric, title in [("classification", "accuracy", "Accuracy by training size"),
                            ("classification", "roc_auc", "ROC-AUC by training size"),
                            ("regression", "r2", "R² by training size")]:
    figure = viz.plot_sample_size_curves(studies[task], metric, title)
    viz.save_figure(figure, OUTPUT_DIR / f"{task}_sample_size_{metric}.png")
    plt.show()"""),
        new_code_cell("""def study_table(study, metric):
    completed = study[study["status"] == "ok"]
    return completed.pivot_table(index="n_train", columns="model", values=metric, aggfunc="mean").round(3)


display(study_table(studies["classification"], "accuracy"))
display(study_table(studies["regression"], "r2"))"""),
        new_markdown_cell("""## Conclusions

The findings below refer to the outputs stored in this notebook (Kumo Tabular `small` on a GTX 980M).

**Where Kumo Tabular performs better**

- Breast cancer classification: best mean on every metric and higher accuracy than the strongest baseline
  (Logistic Regression) on all five splits. The clearest gain is recall on malignant tumours.
- Small training sets in regression: with 25 training rows Kumo reached R² 0.43, against 0.39 for Random Forest
  and 0.08 for Linear Regression, whose estimates are unstable with 10 features and so few rows. Kumo remained the
  best regressor up to 100 rows; at 200 rows it was level with Linear Regression.
- Uncertainty: the regression quantiles give prediction intervals whose 90% nominal coverage was 93% in practice,
  at no extra cost.

**Where it performs worse or offers no advantage**

- Small training sets in classification: Logistic Regression matched Kumo at 25 to 100 rows (0.956 against 0.950
  accuracy at 25 rows). Kumo's classification advantage only appeared from 200 rows onwards. HistGradientBoosting
  predicts a constant at 25 rows because its default minimum leaf size of 20 leaves no room for a split.
- Diabetes regression on the full training set: the mean metrics favour Kumo, but the difference to Linear
  Regression is within the split-to-split noise and the two methods win a similar number of splits.
- Runtime: a single prediction call costs 0.6 to 1.1 s on this GPU and several seconds on CPU, two to three
  orders of magnitude more than Logistic or Linear Regression. Kumo also has to process the full context for
  every new prediction call, whereas a trained classical model predicts almost instantly.
- Dependencies: Kumo needs PyTorch, a model download and, for reasonable speed, a GPU.

**Caveats**

Both datasets are small and well studied, the baselines are untuned and only the `small` checkpoint was evaluated.
These results show how the methods compare out of the box on two classic benchmarks; they are not a general
ranking of tabular methods."""),
    ]
    return new_notebook(cells=cells, metadata=kernel_metadata())


def kernel_metadata() -> dict:
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }


def remove_execution_noise(notebook: nbformat.NotebookNode) -> None:
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        cell.metadata.pop("execution", None)
        if cell.source.startswith("%%writefile"):
            cell.outputs = []


NOTEBOOKS = {
    "01_classification_benchmark.ipynb": classification_notebook,
    "02_regression_benchmark.ipynb": regression_notebook,
    "03_results_analysis.ipynb": analysis_notebook,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Kaggle-ready notebooks from src/.")
    parser.add_argument("--execute", action="store_true", help="run each notebook and store its outputs")
    parser.add_argument("--only", choices=sorted(NOTEBOOKS), help="build a single notebook")
    args = parser.parse_args()

    NOTEBOOK_DIR.mkdir(exist_ok=True)
    for filename, build in NOTEBOOKS.items():
        if args.only and filename != args.only:
            continue
        notebook = build()
        if args.execute:
            print(f"Executing {filename}")
            NotebookClient(notebook, timeout=3600, kernel_name="python3",
                           resources={"metadata": {"path": str(NOTEBOOK_DIR)}}).execute()
            remove_execution_noise(notebook)
        nbformat.write(notebook, NOTEBOOK_DIR / filename)
        print(f"Wrote notebooks/{filename}")


if __name__ == "__main__":
    main()
