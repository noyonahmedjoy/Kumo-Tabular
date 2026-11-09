from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, average_precision_score

MODEL_COLORS = {
    "Kumo Tabular": "#2a78d6",
    "Logistic Regression": "#eb6834",
    "Linear Regression": "#eb6834",
    "Random Forest": "#1baf7a",
    "HistGradientBoosting": "#eda100",
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R²",
}
INK = "#333333"
MUTED = "#8a8a8a"

sns.set_theme(style="whitegrid", context="notebook", rc={
    "axes.edgecolor": "#cccccc",
    "grid.color": "#e6e6e6",
    "axes.titleweight": "bold",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


def color_for(model: str) -> str:
    return MODEL_COLORS.get(model, MUTED)


def save_figure(figure: Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_metric_comparison(summary: pd.DataFrame, metrics: list[str], title: str) -> Figure:
    summary = summary[summary["runs_completed"] > 0]
    models = list(summary["model"])
    figure, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 3.8), squeeze=False)
    for axis, metric in zip(axes[0], metrics):
        means = summary[f"{metric}_mean"].to_numpy()
        stds = summary[f"{metric}_std"].fillna(0).to_numpy()
        positions = np.arange(len(models))
        axis.bar(positions, means, yerr=stds, color=[color_for(m) for m in models], width=0.7,
                 edgecolor="white", linewidth=2, capsize=3, error_kw={"ecolor": INK, "elinewidth": 1})
        low, high = (means - stds).min(), (means + stds).max()
        padding = max((high - low) * 0.35, 1e-3)
        axis.set_ylim(max(low - padding, 0) if low >= 0 else low - padding, high + padding)
        for position, mean in zip(positions, means):
            axis.text(position, mean, f"{mean:.3f}" if abs(mean) < 10 else f"{mean:.1f}",
                      ha="center", va="bottom", fontsize=8, color=INK,
                      bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
        axis.set_title(METRIC_LABELS.get(metric, metric))
        axis.set_xticks(positions, models, rotation=35, ha="right", fontsize=9)
        axis.grid(axis="x", visible=False)
    figure.suptitle(title, y=1.02)
    figure.tight_layout()
    return figure


def plot_confusion_matrices(y_true, predictions: dict[str, np.ndarray], class_names: list[str]) -> Figure:
    figure, axes = plt.subplots(1, len(predictions), figsize=(3.4 * len(predictions), 3.4), squeeze=False)
    for axis, (model, y_pred) in zip(axes[0], predictions.items()):
        matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        sns.heatmap(matrix, annot=True, fmt="d", cbar=False, cmap="Blues", ax=axis, square=True,
                    xticklabels=class_names, yticklabels=class_names, linewidths=2, linecolor="white")
        axis.set_title(model, fontsize=11)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
    figure.tight_layout()
    return figure


def plot_roc_curves(y_true, scores: dict[str, np.ndarray]) -> Figure:
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    for model, score in scores.items():
        fpr, tpr, _ = roc_curve(y_true, score)
        axis.plot(fpr, tpr, linewidth=2, color=color_for(model),
                  label=f"{model} (AUC {roc_auc_score(y_true, score):.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=MUTED)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("ROC curves")
    axis.legend(loc="lower right", fontsize=9, frameon=False)
    return figure


def plot_precision_recall_curves(y_true, scores: dict[str, np.ndarray]) -> Figure:
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    for model, score in scores.items():
        precision, recall, _ = precision_recall_curve(y_true, score)
        axis.step(recall, precision, where="post", linewidth=2, color=color_for(model),
                  label=f"{model} (AP {average_precision_score(y_true, score):.3f})")
    axis.axhline(np.mean(y_true), linestyle="--", linewidth=1, color=MUTED)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-recall curves")
    axis.legend(loc="lower left", fontsize=9, frameon=False)
    return figure


def plot_actual_vs_predicted(y_true, predictions: dict[str, np.ndarray]) -> Figure:
    y_true = np.asarray(y_true)
    figure, axes = plt.subplots(1, len(predictions), figsize=(3.6 * len(predictions), 3.7),
                                sharex=True, sharey=True, squeeze=False)
    limits = [min(y_true.min(), *(p.min() for p in predictions.values())),
              max(y_true.max(), *(p.max() for p in predictions.values()))]
    for axis, (model, y_pred) in zip(axes[0], predictions.items()):
        axis.plot(limits, limits, linestyle="--", linewidth=1, color=MUTED)
        axis.scatter(y_true, y_pred, s=22, alpha=0.75, color=color_for(model), edgecolor="white", linewidth=0.6)
        axis.set_title(model, fontsize=11)
        axis.set_xlabel("Actual")
    axes[0][0].set_ylabel("Predicted")
    figure.tight_layout()
    return figure


def plot_residual_distributions(y_true, predictions: dict[str, np.ndarray]) -> Figure:
    y_true = np.asarray(y_true)
    figure, axes = plt.subplots(1, len(predictions), figsize=(3.6 * len(predictions), 3.4),
                                sharex=True, sharey=True, squeeze=False)
    for axis, (model, y_pred) in zip(axes[0], predictions.items()):
        residuals = y_true - y_pred
        sns.histplot(residuals, bins=20, kde=True, ax=axis, color=color_for(model), edgecolor="white")
        axis.axvline(0, linestyle="--", linewidth=1, color=MUTED)
        axis.set_title(f"{model}\nmean {residuals.mean():.1f}, sd {residuals.std():.1f}", fontsize=10)
        axis.set_xlabel("Actual - predicted")
    figure.tight_layout()
    return figure


def plot_prediction_intervals(y_true, quantiles: pd.DataFrame, point: str, lower: str, upper: str) -> Figure:
    order = np.argsort(quantiles[point].to_numpy())
    y_true = np.asarray(y_true)[order]
    frame = quantiles.iloc[order]
    positions = np.arange(len(order))
    inside = (y_true >= frame[lower].to_numpy()) & (y_true <= frame[upper].to_numpy())
    figure, axis = plt.subplots(figsize=(8.5, 4))
    axis.fill_between(positions, frame[lower], frame[upper], color=color_for("Kumo Tabular"), alpha=0.18,
                      linewidth=0, label=f"{lower}-{upper} interval")
    axis.plot(positions, frame[point], color=color_for("Kumo Tabular"), linewidth=2, label=f"{point} prediction")
    axis.scatter(positions[inside], y_true[inside], s=14, color=INK, label="Actual (inside)")
    axis.scatter(positions[~inside], y_true[~inside], s=22, marker="x", color="#d03b3b", label="Actual (outside)")
    axis.set_xlabel("Test rows sorted by predicted value")
    axis.set_ylabel("Disease progression")
    axis.set_title(f"Kumo Tabular prediction intervals (coverage {inside.mean():.1%})")
    axis.legend(fontsize=9, frameon=False, loc="upper left")
    return figure


def plot_inference_time(summary: pd.DataFrame, title: str) -> Figure:
    summary = summary[summary["runs_completed"] > 0].sort_values("total_seconds_mean")
    figure, axis = plt.subplots(figsize=(6.5, 0.55 * len(summary) + 1.2))
    bars = axis.barh(summary["model"], summary["total_seconds_mean"],
                     color=[color_for(m) for m in summary["model"]], edgecolor="white", linewidth=2, height=0.65)
    axis.set_xscale("log")
    axis.bar_label(bars, labels=[f"{value:.3f}s" for value in summary["total_seconds_mean"]],
                   padding=4, fontsize=9, color=INK)
    axis.set_xlabel("Fit + predict time per split (seconds, log scale)")
    axis.set_title(title)
    axis.grid(axis="y", visible=False)
    axis.set_xlim(right=summary["total_seconds_mean"].max() * 4)
    return figure


def plot_sample_size_curves(study: pd.DataFrame, metric: str, title: str) -> Figure:
    completed = study[study["status"] == "ok"]
    grouped = completed.groupby(["model", "n_train"])[metric].agg(["mean", "std"]).reset_index()
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    for model in dict.fromkeys(completed["model"]):
        rows = grouped[grouped["model"] == model]
        axis.errorbar(rows["n_train"], rows["mean"], yerr=rows["std"].fillna(0), marker="o", markersize=6,
                      linewidth=2, capsize=3, color=color_for(model), label=model)
    axis.set_xscale("log")
    axis.set_xticks(sorted(grouped["n_train"].unique()), [str(n) for n in sorted(grouped["n_train"].unique())])
    axis.minorticks_off()
    axis.set_xlabel("Training rows (Kumo context size)")
    axis.set_ylabel(METRIC_LABELS.get(metric, metric))
    axis.set_title(title)
    axis.legend(fontsize=9, frameon=False)
    return figure


def save_classification_figures(
    run, summary: pd.DataFrame, study: pd.DataFrame, figures_dir: str | Path
) -> list[Path]:
    figures_dir = Path(figures_dir)
    figures = {
        "classification_metrics.png": plot_metric_comparison(
            summary, ["accuracy", "precision", "recall", "f1", "roc_auc"],
            "Breast cancer classification: mean and standard deviation across splits"),
        "classification_confusion_matrices.png": plot_confusion_matrices(
            run.split.y_test, run.predictions, ["benign", "malignant"]),
        "classification_roc_curves.png": plot_roc_curves(run.split.y_test, run.scores),
        "classification_precision_recall.png": plot_precision_recall_curves(run.split.y_test, run.scores),
        "classification_inference_time.png": plot_inference_time(summary, "Classification runtime"),
        "classification_sample_size_accuracy.png": plot_sample_size_curves(
            study, "accuracy", "Accuracy by training size"),
    }
    paths = [save_figure(figure, figures_dir / name) for name, figure in figures.items()]
    plt.close("all")
    return paths


def save_regression_figures(
    run, summary: pd.DataFrame, study: pd.DataFrame, kumo_config: dict, figures_dir: str | Path
) -> list[Path]:
    figures_dir = Path(figures_dir)
    figures = {
        "regression_metrics.png": plot_metric_comparison(
            summary, ["mae", "rmse", "r2"], "Diabetes regression: mean and standard deviation across splits"),
        "regression_actual_vs_predicted.png": plot_actual_vs_predicted(run.split.y_test, run.predictions),
        "regression_residuals.png": plot_residual_distributions(run.split.y_test, run.predictions),
        "regression_inference_time.png": plot_inference_time(summary, "Regression runtime"),
        "regression_sample_size_r2.png": plot_sample_size_curves(study, "r2", "R² by training size"),
    }
    if run.kumo_quantiles is not None:
        lower, upper = kumo_config["regression_interval"]
        figures["regression_kumo_intervals.png"] = plot_prediction_intervals(
            run.split.y_test, run.kumo_quantiles, kumo_config["regression_point_quantile"], lower, upper)
    paths = [save_figure(figure, figures_dir / name) for name, figure in figures.items()]
    plt.close("all")
    return paths
