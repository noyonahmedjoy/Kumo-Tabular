import json
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization import INK, MUTED, save_figure  # noqa: E402

RESULTS = PROJECT_ROOT / "results" / "openml_benchmark"
OUTPUT = RESULTS / "analysis"
KUMO = "Kumo Tabular"
PRIMARY = {"classification": "log_loss", "regression": "rmse"}
METRICS = {
    "classification": {"log_loss": False, "accuracy": True, "roc_auc": True, "macro_f1": True},
    "regression": {"rmse": False, "mae": False, "r2": True},
}
OVERLAP_WITH_ORIGINAL_STUDY = {1510}
FAMILIES = {1049: "nasa_defect", 1050: "nasa_defect", 1063: "nasa_defect", 1068: "nasa_defect",
            18: "mfeat", 22: "mfeat"}
COLORS = {KUMO: "#2a78d6"}


def holm(p_values: dict) -> dict:
    order = sorted(p_values, key=p_values.get)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted


def dataset_means(results: pd.DataFrame, task: str) -> pd.DataFrame:
    rows = results[(results["task"] == task)]
    failed = rows[rows["status"] != "ok"]
    if len(failed):
        raise SystemExit(f"{len(failed)} failed or incomplete rows for {task}; analysis requires complete results")
    folds = rows.groupby(["data_id", "model"]).size()
    if folds.nunique() != 1:
        raise SystemExit(f"unequal fold counts for {task}: {folds.value_counts().to_dict()}")
    return rows.groupby(["data_id", "dataset", "model"])[list(METRICS[task]) + ["total_seconds"]].mean().reset_index()


def ranks(means: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.DataFrame:
    table = means.pivot(index="data_id", columns="model", values=metric)
    return table.rank(axis=1, ascending=not higher_is_better, method="average")


def collapse_families(table: pd.DataFrame) -> pd.DataFrame:
    keys = [FAMILIES.get(int(i), str(i)) for i in table.index]
    return table.groupby(keys).mean()


def compare(table: pd.DataFrame, higher_is_better: bool) -> pd.DataFrame:
    rows, p_values = [], {}
    for model in [m for m in table.columns if m != KUMO]:
        advantage = (table[KUMO] - table[model]) if higher_is_better else (table[model] - table[KUMO])
        relative = advantage / table[model].abs()
        nonzero = advantage[advantage.abs() > 1e-12]
        p = float(stats.wilcoxon(nonzero, alternative="two-sided").pvalue) if len(nonzero) >= 1 else 1.0
        p_values[model] = p
        rows.append({"baseline": model, "datasets": len(advantage), "kumo_wins": int((advantage > 1e-12).sum()),
                     "ties": int((advantage.abs() <= 1e-12).sum()), "kumo_losses": int((advantage < -1e-12).sum()),
                     "median_advantage": float(advantage.median()),
                     "median_relative_advantage": float(relative.median()), "wilcoxon_p": p})
    adjusted = holm(p_values)
    frame = pd.DataFrame(rows)
    frame["holm_p"] = frame["baseline"].map(adjusted)
    return frame


def analyse(means: pd.DataFrame, task: str, exclude: set | None = None, families: bool = False) -> dict:
    data = means[~means["data_id"].isin(exclude or set())]
    result = {}
    for metric, higher in METRICS[task].items():
        table = data.pivot(index="data_id", columns="model", values=metric)
        if families:
            table = collapse_families(table)
        rank_table = table.rank(axis=1, ascending=not higher, method="average")
        friedman = stats.friedmanchisquare(*[table[c] for c in table.columns])
        result[metric] = {
            "datasets": int(len(table)),
            "average_rank": rank_table.mean().sort_values().round(3).to_dict(),
            "friedman_statistic": float(friedman.statistic),
            "friedman_p": float(friedman.pvalue),
            "kumo_vs_baselines": compare(table, higher).to_dict(orient="records"),
        }
    return result


def rank_figure(analysis: dict) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for axis, task in zip(axes, ("classification", "regression")):
        metric = PRIMARY[task]
        average = pd.Series(analysis[task]["all"][metric]["average_rank"]).sort_values()
        colors = [COLORS.get(m, MUTED) for m in average.index]
        axis.barh(average.index[::-1], average.values[::-1], color=colors[::-1], height=0.65,
                  edgecolor="white", linewidth=2)
        for y, value in enumerate(average.values[::-1]):
            axis.text(value + 0.05, y, f"{value:.2f}", va="center", fontsize=9, color=INK)
        n = analysis[task]["all"][metric]["datasets"]
        p = analysis[task]["all"][metric]["friedman_p"]
        axis.set_title(f"{task.capitalize()}: average rank by {metric.upper().replace('_', ' ')}\n"
                       f"({n} datasets; Friedman p = {p:.2g})", fontsize=10)
        axis.set_xlabel("Average rank (1 = best)")
        axis.set_xlim(1, len(average) + 0.6)
        axis.grid(axis="y", visible=False)
    figure.tight_layout()
    return figure


def per_dataset_figure(means: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [15, 7]})
    for axis, task in zip(axes, ("classification", "regression")):
        metric = PRIMARY[task]
        table = means[means["task_type"] == task].pivot(index="dataset", columns="model", values=metric)
        best_baseline = table.drop(columns=KUMO).min(axis=1)
        relative = ((best_baseline - table[KUMO]) / best_baseline * 100).sort_values()
        colors = [COLORS[KUMO] if v > 0 else MUTED for v in relative]
        axis.barh(relative.index, relative.values, color=colors, height=0.7)
        axis.axvline(0, color=INK, linewidth=1)
        axis.set_xlabel(f"Kumo vs best baseline, % lower {metric.upper().replace('_', ' ')}")
        axis.set_title(f"{task.capitalize()} (positive favours Kumo)", fontsize=10)
        axis.tick_params(axis="y", labelsize=8)
        axis.grid(axis="y", visible=False)
    figure.tight_layout()
    return figure


def runtime_figure(results: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for axis, task in zip(axes, ("classification", "regression")):
        rows = results[(results["task"] == task) & (results["status"] == "ok")]
        order = rows.groupby("model")["total_seconds"].median().sort_values().index
        data = [rows.loc[rows["model"] == m, "total_seconds"].to_numpy() for m in order]
        box = axis.boxplot(data, vert=False, tick_labels=list(order), showfliers=False, patch_artist=True)
        for patch, model in zip(box["boxes"], order):
            patch.set_facecolor(COLORS.get(model, "#e6e6e6"))
        axis.set_xscale("log")
        axis.set_xlabel("Fit + predict seconds per fold (log scale)")
        axis.set_title(f"{task.capitalize()} runtime (Kumo on GPU, others on CPU)", fontsize=10)
    figure.tight_layout()
    return figure


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(RESULTS / "results_by_fold.csv")
    analysis, all_means = {}, []
    for task in ("classification", "regression"):
        means = dataset_means(results, task)
        means["task_type"] = task
        all_means.append(means)
        analysis[task] = {
            "all": analyse(means, task),
            "excluding_overlap_with_original_study": analyse(means, task, exclude=OVERLAP_WITH_ORIGINAL_STUDY),
            "dataset_families_collapsed": analyse(means, task, families=True),
        }
        runtime = results[(results["task"] == task) & (results["status"] == "ok")].groupby("model")["total_seconds"]
        analysis[task]["runtime_seconds_per_fold"] = {
            "median": runtime.median().round(4).to_dict(),
            "q25": runtime.quantile(0.25).round(4).to_dict(),
            "q75": runtime.quantile(0.75).round(4).to_dict(),
        }
    means = pd.concat(all_means, ignore_index=True)
    means.to_csv(OUTPUT / "dataset_means.csv", index=False)
    (OUTPUT / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    save_figure(rank_figure(analysis), OUTPUT / "average_ranks.pdf")
    save_figure(per_dataset_figure(means), OUTPUT / "per_dataset_primary_metric.pdf")
    save_figure(runtime_figure(results), OUTPUT / "runtime.pdf")
    plt.close("all")
    for task in ("classification", "regression"):
        block = analysis[task]["all"][PRIMARY[task]]
        print(f"== {task}: {block['datasets']} datasets, primary metric {PRIMARY[task]}, "
              f"Friedman p = {block['friedman_p']:.3g}")
        print("   average ranks:", block["average_rank"])
        for row in block["kumo_vs_baselines"]:
            print(f"   vs {row['baseline']:<22} W/T/L {row['kumo_wins']}/{row['ties']}/{row['kumo_losses']} "
                  f"median rel. adv. {row['median_relative_advantage']:+.3f}  p={row['wilcoxon_p']:.3g}  "
                  f"Holm p={row['holm_p']:.3g}")


if __name__ == "__main__":
    main()
