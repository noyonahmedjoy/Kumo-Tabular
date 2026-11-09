import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

PAPER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PAPER_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking import create_kumo_predictor, load_config, run_benchmark  # noqa: E402
from src.data_loader import load_dataset, split_dataset  # noqa: E402
from src.visualization import (  # noqa: E402
    INK, MUTED, color_for, plot_precision_recall_curves, plot_sample_size_curves, save_figure,
)

RESULTS = PROJECT_ROOT / "results"
SOURCE = PAPER_DIR / "final_submission" / "latex_source"
TABLES = SOURCE / "tables"
FIGURES = SOURCE / "figures"
PAPER_FIGURES = [
    "classification_confusion_matrices.png", "classification_roc_curves.png",
    "regression_actual_vs_predicted.png", "regression_residuals.png", "regression_kumo_intervals.png",
    "regression_sample_size_r2.png",
    "classification_inference_time.png", "regression_inference_time.png",
]
KUMO = "Kumo Tabular"
TASKS = {
    "classification": {"baseline": "Logistic Regression", "metrics": {
        "accuracy": True, "precision": True, "recall": True, "f1": True, "roc_auc": True}},
    "regression": {"baseline": "Linear Regression", "metrics": {"mae": False, "rmse": False, "r2": True}},
}
LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC",
          "mae": "MAE", "rmse": "RMSE", "r2": "$R^2$"}
SHORT = {"Kumo Tabular": "Kumo Tabular", "Logistic Regression": "Logistic Reg.", "Linear Regression": "Linear Reg.",
         "Random Forest": "Random Forest", "HistGradientBoosting": "HistGB"}


def paired_tests(kumo: np.ndarray, baseline: np.ndarray, higher_is_better: bool, test_fraction: float) -> dict:
    advantage = kumo - baseline if higher_is_better else baseline - kumo
    k = len(advantage)
    mean, sd = advantage.mean(), advantage.std(ddof=1)
    critical = stats.t.ppf(0.975, k - 1)
    naive_se = sd / np.sqrt(k)
    corrected_se = sd * np.sqrt(1 / k + test_fraction)
    nonzero = advantage[np.abs(advantage) > 1e-12]
    return {
        "advantage": advantage.tolist(),
        "mean_advantage": mean,
        "sd_advantage": sd,
        "wins": int((advantage > 1e-12).sum()),
        "ties": int((np.abs(advantage) <= 1e-12).sum()),
        "losses": int((advantage < -1e-12).sum()),
        "naive_p": float(2 * stats.t.sf(abs(mean / naive_se), k - 1)),
        "corrected_ci": [mean - critical * corrected_se, mean + critical * corrected_se],
        "corrected_p": float(2 * stats.t.sf(abs(mean / corrected_se), k - 1)),
        "wilcoxon_p": float(stats.wilcoxon(nonzero, method="exact").pvalue),
        "wilcoxon_n": int(len(nonzero)),
    }


def holm(p_values: dict) -> dict:
    order = sorted(p_values, key=p_values.get)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace("-", "$-$")


def bold_best(values: list[float], higher_is_better: bool) -> list[bool]:
    best = max(values) if higher_is_better else min(values)
    return [abs(v - best) < 1e-12 for v in values]


def results_table(summary: pd.DataFrame, metrics: dict, digits: dict, label: str, caption: str) -> str:
    columns = "l" + "c" * (len(metrics) + 1)
    header = " & ".join(["Model", *[LABELS[m] for m in metrics], "Time (s)"])
    lines = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\setlength{\tabcolsep}{5pt}", rf"\begin{{tabular}}{{{columns}}}", r"\toprule", header + r" \\",
             r"\midrule"]
    best = {m: bold_best(summary[f"{m}_mean"].tolist(), h) for m, h in metrics.items()}
    for i, row in summary.iterrows():
        cells = [SHORT[row["model"]]]
        for m in metrics:
            text = rf"{fmt(row[f'{m}_mean'], digits[m])} $\pm$ {fmt(row[f'{m}_std'], digits[m])}"
            cells.append(rf"\textbf{{{text}}}" if best[m][i] else text)
        cells.append(rf"{fmt(row['total_seconds_mean'], 3)} $\pm$ {fmt(row['total_seconds_std'], 3)}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def per_seed_table(results: dict) -> str:
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Per-split results of Kumo Tabular and the strongest baseline. Seeds define the "
             r"train/test partition shared by all models.}",
             r"\label{tab:per_split}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{ccccc}", r"\toprule",
             r"& \multicolumn{2}{c}{Accuracy} & \multicolumn{2}{c}{$R^2$} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Seed & Kumo & Logistic Reg. & Kumo & Linear Reg. \\", r"\midrule"]
    acc = results["classification"].pivot(index="seed", columns="model", values="accuracy")
    r2 = results["regression"].pivot(index="seed", columns="model", values="r2")
    for seed in acc.index:
        lines.append(f"{seed} & {fmt(acc.loc[seed, KUMO])} & {fmt(acc.loc[seed, 'Logistic Regression'])} & "
                     f"{fmt(r2.loc[seed, KUMO])} & {fmt(r2.loc[seed, 'Linear Regression'])}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def paired_table(paired: dict) -> str:
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Paired comparison of Kumo Tabular with the strongest baseline over the five splits. "
             r"$\overline{d}$ is the mean advantage of Kumo (positive favours Kumo for every metric), $s_d$ its standard "
             r"deviation. The corrected test and interval use the Nadeau--Bengio variance correction for "
             r"overlapping training sets; Holm adjustment is applied within each task. No result is significant "
             r"at $\alpha=0.05$ after adjustment.}",
             r"\label{tab:paired}", r"\setlength{\tabcolsep}{4.5pt}", r"\begin{tabular}{llcccccccc}", r"\toprule",
             r"Task & Metric & W/T/L & $\overline{d}$ & $s_d$ & Corrected 95\% CI & $p_{\mathrm{naive}}$ & "
             r"$p_{\mathrm{corr}}$ & $p_{\mathrm{corr}}^{\mathrm{Holm}}$ & $p_{\mathrm{Wilcoxon}}$ \\", r"\midrule"]
    for task, rows in paired.items():
        adjusted = holm({m: r["corrected_p"] for m, r in rows.items()})
        digits = 4 if task == "classification" else 3
        for index, (metric, r) in enumerate(rows.items()):
            name = ("Classification" if task == "classification" else "Regression") if index == 0 else ""
            d = 2 if metric in ("mae", "rmse") else digits
            ci = rf"[{fmt(r['corrected_ci'][0], d)}, {fmt(r['corrected_ci'][1], d)}]"
            lines.append(" & ".join([
                name, LABELS[metric], f"{r['wins']}/{r['ties']}/{r['losses']}", fmt(r["mean_advantage"], d),
                fmt(r["sd_advantage"], d), ci, fmt(r["naive_p"], 3), fmt(r["corrected_p"], 3),
                fmt(adjusted[metric], 3), f"{r['wilcoxon_p']:.3f} ($n$={r['wilcoxon_n']})"]) + r" \\")
            r["corrected_p_holm"] = adjusted[metric]
        if task == "classification":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def study_table(study: pd.DataFrame, metric: str, models: list[str], label: str, caption: str, higher: bool) -> str:
    completed = study[study["status"] == "ok"]
    mean = completed.pivot_table(index="n_train", columns="model", values=metric, aggfunc="mean")
    sd = completed.pivot_table(index="n_train", columns="model", values=metric, aggfunc="std")
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{rcccc}", r"\toprule",
             "$n$ & " + " & ".join(SHORT[m] for m in models) + r" \\", r"\midrule"]
    for n in mean.index:
        values = [mean.loc[n, m] for m in models]
        best = bold_best(values, higher)
        cells = []
        for m, v, b in zip(models, values, best):
            text = rf"{fmt(v)}\,{{\scriptsize$\pm${fmt(sd.loc[n, m])}}}"
            cells.append(rf"\textbf{{{fmt(v)}}}\,{{\scriptsize$\pm${fmt(sd.loc[n, m])}}}" if b else text)
        lines.append(f"{n} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def paired_figure(paired: dict) -> plt.Figure:
    groups = [
        ("Classification", ["accuracy", "precision", "recall", "f1", "roc_auc"], "Kumo advantage"),
        ("Regression error", ["mae", "rmse"], "Kumo advantage (target units)"),
        ("Regression fit", ["r2"], "Kumo advantage"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), gridspec_kw={"width_ratios": [5, 2.2, 1.4]})
    rng = np.random.default_rng(0)
    for axis, (title, metrics, ylabel) in zip(axes, groups):
        task = "classification" if title == "Classification" else "regression"
        for position, metric in enumerate(metrics):
            row = paired[task][metric]
            jitter = rng.uniform(-0.12, 0.12, len(row["advantage"]))
            axis.scatter(position + jitter, row["advantage"], s=22, color=MUTED, edgecolor="white", linewidth=0.5,
                         zorder=2, label="Single split" if position == 0 else None)
            low, high = row["corrected_ci"]
            axis.errorbar(position + 0.32, row["mean_advantage"],
                          yerr=[[row["mean_advantage"] - low], [high - row["mean_advantage"]]],
                          fmt="o", color=color_for(KUMO), markersize=6, capsize=3, linewidth=2, zorder=3,
                          label="Mean, corrected 95% CI" if position == 0 else None)
        axis.axhline(0, color=INK, linewidth=1)
        axis.set_xticks(range(len(metrics)), [LABELS[m].replace("$R^2$", "R²") for m in metrics])
        axis.set_xlim(-0.5, len(metrics) - 0.3)
        axis.set_title(title, fontsize=11)
        axis.set_ylabel(ylabel)
        axis.grid(axis="x", visible=False)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    figure.tight_layout()
    return figure


def redrawn_figures(studies: dict, results: dict) -> None:
    figure = plot_sample_size_curves(studies["classification"], "accuracy", "Accuracy by training size")
    ticks = [25, 50, 100, 200, 455]
    figure.axes[0].set_xticks(ticks, [str(t) for t in ticks])
    save_figure(figure, FIGURES / "classification_sample_size_accuracy.png")

    config = load_config(PROJECT_ROOT / "configs" / "experiment_config.yaml")
    kumo, error = create_kumo_predictor("classification", config["kumo"])
    if kumo is None:
        raise RuntimeError(f"Kumo Tabular is required to redraw the precision-recall figure: {error}")
    name, features, target = load_dataset("classification")
    seed = config["seeds"][0]
    split = split_dataset(features, target, task="classification", name=name, test_size=config["test_size"],
                          random_state=seed)
    run = run_benchmark(split, config, kumo)
    saved = results["classification"][results["classification"]["seed"] == seed].set_index("model")
    for _, row in run.results.iterrows():
        for metric in TASKS["classification"]["metrics"]:
            if abs(row[metric] - saved.loc[row["model"], metric]) > 1e-12:
                raise RuntimeError(f"Recomputed {metric} for {row['model']} differs from results/")
    figure = plot_precision_recall_curves(split.y_test, run.scores)
    figure.axes[0].legend(loc="center left", fontsize=9, frameon=False)
    save_figure(figure, FIGURES / "classification_precision_recall.png")
    plt.close("all")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name in PAPER_FIGURES:
        shutil.copy2(RESULTS / "figures" / name, FIGURES / name)
    results = {task: pd.read_csv(RESULTS / task / "results_by_seed.csv") for task in TASKS}
    summaries = {task: pd.read_csv(RESULTS / task / "summary.csv") for task in TASKS}
    studies = {task: pd.read_csv(RESULTS / task / "sample_size_study.csv") for task in TASKS}

    paired = {}
    for task, spec in TASKS.items():
        completed = results[task][results[task]["status"] == "ok"]
        test_fraction = completed["n_test"].iloc[0] / completed["n_train"].iloc[0]
        paired[task] = {}
        for metric, higher in spec["metrics"].items():
            by_seed = completed.pivot(index="seed", columns="model", values=metric)
            paired[task][metric] = paired_tests(
                by_seed[KUMO].to_numpy(), by_seed[spec["baseline"]].to_numpy(), higher, test_fraction)

    tables = {
        "table_classification.tex": results_table(
            summaries["classification"], TASKS["classification"]["metrics"],
            dict.fromkeys(TASKS["classification"]["metrics"], 3), "tab:classification",
            "Breast cancer classification results (mean $\\pm$ standard deviation over five stratified splits, "
            "114 test rows each). Malignant is the positive class. Time is fit plus prediction time per split. "
            "Best mean per metric in bold."),
        "table_regression.tex": results_table(
            summaries["regression"], TASKS["regression"]["metrics"], {"mae": 2, "rmse": 2, "r2": 3},
            "tab:regression",
            "Diabetes regression results (mean $\\pm$ standard deviation over five random splits, 89 test rows "
            "each). Kumo Tabular uses the median quantile $q_{0.5}$ as point prediction. Best mean per metric "
            "in bold."),
        "table_per_split.tex": per_seed_table(results),
        "table_paired.tex": paired_table(paired),
        "table_study_classification.tex": study_table(
            studies["classification"], "accuracy",
            [KUMO, "Logistic Regression", "Random Forest", "HistGradientBoosting"], "tab:study_classification",
            "Classification accuracy by training-set size (mean $\\pm$ standard deviation over three seeds; "
            "full test split).", True),
        "table_study_regression.tex": study_table(
            studies["regression"], "r2", [KUMO, "Linear Regression", "Random Forest", "HistGradientBoosting"],
            "tab:study_regression",
            "Regression $R^2$ by training-set size (mean $\\pm$ standard deviation over three seeds; full test "
            "split).", True),
    }
    for name, content in tables.items():
        (TABLES / name).write_text(content + "\n", encoding="utf-8")

    save_figure(paired_figure(paired), FIGURES / "paired_differences.pdf")
    plt.close("all")
    redrawn_figures(studies, results)

    statistics = {
        "paired": paired,
        "confusion_totals": results["classification"].groupby("model")[["tn", "fp", "fn", "tp"]].sum()
        .to_dict(orient="index"),
        "kumo_seconds_by_size": {
            task: studies[task][studies[task]["model"] == KUMO].groupby("n_train")["predict_seconds"].mean().to_dict()
            for task in TASKS},
    }
    (PAPER_DIR / "statistics.json").write_text(json.dumps(statistics, indent=2, default=float), encoding="utf-8")
    for task, rows in paired.items():
        for metric, r in rows.items():
            print(f"{task:14s} {metric:9s} W/T/L={r['wins']}/{r['ties']}/{r['losses']} d={r['mean_advantage']:+.4f} "
                  f"p_corr={r['corrected_p']:.4f} holm={r['corrected_p_holm']:.4f} wilcoxon={r['wilcoxon_p']:.4f}")


if __name__ == "__main__":
    main()
