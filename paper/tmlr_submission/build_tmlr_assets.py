import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "results" / "openml_benchmark" / "analysis"
MANUSCRIPT = Path(__file__).resolve().parent / "manuscript"
TABLES = MANUSCRIPT / "tables"
FIGURES = MANUSCRIPT / "figures"
IEEE = ROOT / "paper" / "final_submission" / "latex_source"
KUMO = "Kumo Tabular"
PRIMARY = {"classification": "log_loss", "regression": "rmse"}
NAMES = {"log_loss": "log loss", "rmse": "RMSE", "accuracy": "accuracy", "roc_auc": "ROC-AUC", "macro_f1": "macro-F1",
         "mae": "MAE", "r2": "$R^2$"}
SHORT = {"Kumo Tabular": "Kumo Tabular", "Logistic Regression": "Logistic reg.", "Linear Regression": "Linear reg.",
         "Random Forest": "Random forest", "Extra Trees": "Extra trees", "HistGradientBoosting": "HistGB",
         "XGBoost": "XGBoost", "LightGBM": "LightGBM", "CatBoost": "CatBoost"}


def tex(text: str) -> str:
    return str(text).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def p_fmt(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write(name: str, lines: list[str]) -> None:
    (TABLES / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def fit_to_width(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if r"\begin{adjustbox}" not in text:
        text = text.replace(r"\begin{tabular}", "\\smallskip\n\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}", 1)
        text = text.replace(r"\end{tabular}", "\\end{tabular}\n\\end{adjustbox}", 1)
        path.write_text(text, encoding="utf-8")


def dataset_table(suite: dict) -> None:
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Datasets of the OpenML benchmark, selected from OpenML-CC18 and OpenML-CTR23 by fixed "
             r"criteria (150--2{,}000 rows, at most 60 features, numerical features only, no missing values, at "
             r"most 10 classes). \texttt{forest\_fires} (OpenML 44962) met the criteria according to its OpenML metadata but contains "
             r"two text columns and was excluded.}", r"\label{tab:datasets}", r"\begin{tabular}{rlrrr|rlrr}", r"\toprule",
             r"\multicolumn{5}{c|}{Classification} & \multicolumn{4}{c}{Regression} \\",
             r"ID & Name & Rows & Feat. & Cls. & ID & Name & Rows & Feat. \\", r"\midrule"]
    excluded = {item["data_id"] for item in suite.get("excluded_at_load", [])}
    cls = [e for e in suite["classification"] if e["data_id"] not in excluded]
    reg = [e for e in suite["regression"] if e["data_id"] not in excluded]
    for i in range(max(len(cls), len(reg))):
        left = (f"{cls[i]['data_id']} & {tex(cls[i]['name'])} & {cls[i]['rows']} & {cls[i]['features']} & "
                f"{cls[i]['classes']}") if i < len(cls) else "& & & &"
        right = (f"{reg[i]['data_id']} & {tex(reg[i]['name'])} & {reg[i]['rows']} & {reg[i]['features']}"
                 if i < len(reg) else "& & &")
        lines.append(f"{left} & {right} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("datasets.tex", lines)


def comparison_table(analysis: dict) -> None:
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Kumo Tabular versus each baseline on the primary metric, with datasets as the unit of "
             r"analysis. W/T/L: datasets on which Kumo is better/equal/worse. Relative advantage: median over "
             r"datasets of the baseline's loss minus Kumo's loss, divided by the baseline's loss (positive favours "
             r"Kumo). $p$: two-sided Wilcoxon signed-rank test; Holm: adjusted across the seven baselines of a task. "
             r"Avg.\ rank: average rank among all eight models (1 = best).}",
             r"\label{tab:comparison}", r"\begin{tabular}{llcrrrr}", r"\toprule",
             r"Task (metric) & Baseline & W/T/L & Rel.\ adv. & $p$ & Holm $p$ & Avg.\ rank \\", r"\midrule"]
    for task in ("classification", "regression"):
        block = analysis[task]["all"][PRIMARY[task]]
        ranks = block["average_rank"]
        rows = sorted(block["kumo_vs_baselines"], key=lambda r: ranks[r["baseline"]])
        label = f"{task.capitalize()} ({NAMES[PRIMARY[task]]})"
        lines.append(f"\\multirow{{{len(rows) + 1}}}{{*}}{{\\shortstack[l]{{{label}\\\\$n={block['datasets']}$}}}} "
                     f"& {KUMO} & -- & -- & -- & -- & {ranks[KUMO]:.2f} \\\\")
        for r in rows:
            lines.append(f"& {SHORT[r['baseline']]} & {r['kumo_wins']}/{r['ties']}/{r['kumo_losses']} & "
                         f"{100 * r['median_relative_advantage']:+.1f}\\% & {p_fmt(r['wilcoxon_p'])} & "
                         f"{p_fmt(r['holm_p'])} & {ranks[r['baseline']]:.2f} \\\\")
        lines.append(r"\midrule" if task == "classification" else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table}"]
    write("comparison.tex", lines)


def secondary_table(analysis: dict) -> None:
    models = list(SHORT)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Average rank of each model per metric (1 = best) and Friedman test $p$-value. Primary "
             r"metrics in the first column of each task; the others are descriptive.}",
             r"\label{tab:ranks}", r"\begin{tabular}{lcccc|ccc}", r"\toprule",
             rf"& \multicolumn{{4}}{{c|}}{{Classification ($n={analysis['classification']['all']['log_loss']['datasets']}$)}} & "
             rf"\multicolumn{{3}}{{c}}{{Regression ($n={analysis['regression']['all']['rmse']['datasets']}$)}} \\",
             r"Model & log loss & accuracy & ROC-AUC & macro-F1 & RMSE & MAE & $R^2$ \\", r"\midrule"]
    cls = analysis["classification"]["all"]
    reg = analysis["regression"]["all"]
    for model in models:
        values = []
        for metric in ("log_loss", "accuracy", "roc_auc", "macro_f1"):
            values.append(f"{cls[metric]['average_rank'][model]:.2f}" if model in cls[metric]["average_rank"] else "--")
        for metric in ("rmse", "mae", "r2"):
            values.append(f"{reg[metric]['average_rank'][model]:.2f}" if model in reg[metric]["average_rank"] else "--")
        lines.append(f"{SHORT[model]} & " + " & ".join(values) + r" \\")
    lines.append(r"\midrule")
    friedman = [p_fmt(cls[m]["friedman_p"]) for m in ("log_loss", "accuracy", "roc_auc", "macro_f1")]
    friedman += [p_fmt(reg[m]["friedman_p"]) for m in ("rmse", "mae", "r2")]
    lines.append(r"Friedman $p$ & " + " & ".join(friedman) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("ranks.tex", lines)


def sensitivity_table(analysis: dict) -> None:
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Planned sensitivity analyses on the primary metric: Kumo Tabular's average rank, the "
             r"number of baselines it beats with Holm-adjusted $p<0.05$, and its W/T/L against the strongest "
             r"baseline (lowest average rank).}",
             r"\label{tab:sensitivity}", r"\begin{tabular}{llcccc}", r"\toprule",
             r"Task & Analysis & $n$ & Kumo avg.\ rank & Sig.\ wins (of 7) & W/T/L vs strongest \\", r"\midrule"]
    labels = {"all": "All datasets", "excluding_overlap_with_original_study": "Excluding wdbc",
              "dataset_families_collapsed": "Families collapsed"}
    for task in ("classification", "regression"):
        for key, label in labels.items():
            block = analysis[task][key][PRIMARY[task]]
            ranks = block["average_rank"]
            strongest = min((m for m in ranks if m != KUMO), key=ranks.get)
            row = next(r for r in block["kumo_vs_baselines"] if r["baseline"] == strongest)
            significant = sum(1 for r in block["kumo_vs_baselines"]
                              if r["holm_p"] < 0.05 and r["kumo_wins"] > r["kumo_losses"])
            lines.append(f"{task.capitalize() if key == 'all' else ''} & {label} & {block['datasets']} & "
                         f"{ranks[KUMO]:.2f} & {significant} & {row['kumo_wins']}/{row['ties']}/{row['kumo_losses']} "
                         f"({SHORT[strongest]}) \\\\")
        lines.append(r"\midrule" if task == "classification" else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table}"]
    write("sensitivity.tex", lines)


def runtime_table(analysis: dict) -> None:
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Fit plus prediction time per fold in seconds: median [interquartile range] over all folds "
             r"of all datasets. Kumo Tabular ran on an NVIDIA GTX 980M GPU (float32); all other models on an Intel "
             r"Core i7-6700HQ CPU. Times are hardware-dependent and not comparable as algorithmic costs.}",
             r"\label{tab:runtime}", r"\begin{tabular}{lll}", r"\toprule",
             r"Model & Classification & Regression \\", r"\midrule"]
    cls = analysis["classification"]["runtime_seconds_per_fold"]
    reg = analysis["regression"]["runtime_seconds_per_fold"]
    for model in SHORT:
        cell = lambda block: (f"{block['median'][model]:.3f} [{block['q25'][model]:.3f}, {block['q75'][model]:.3f}]"
                              if model in block["median"] else "--")
        lines.append(f"{SHORT[model]} & {cell(cls)} & {cell(reg)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("runtime.tex", lines)


def per_dataset_tables(means: pd.DataFrame) -> None:
    for task in ("classification", "regression"):
        metric = PRIMARY[task]
        table = means[means["task_type"] == task].pivot(index="dataset", columns="model", values=metric)
        models = [m for m in SHORT if m in table.columns]
        header = " & ".join(SHORT[m] for m in models)
        lines = [r"\begin{table}[h]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
                 rf"\caption{{Per-dataset {NAMES[metric]} ({task}), mean over 10 folds. Best value per dataset in "
                 rf"bold.}}", rf"\label{{tab:perdataset_{task}}}", r"\begin{tabular}{l" + "r" * len(models) + "}",
                 r"\toprule", f"Dataset & {header} \\\\", r"\midrule"]
        for dataset, row in table[models].iterrows():
            best = row.min()
            cells = [rf"\textbf{{{v:.3f}}}" if abs(v - best) < 1e-12 else f"{v:.3f}" for v in row]
            lines.append(f"{tex(dataset)} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        write(f"per_dataset_{task}.tex", lines)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    analysis = json.loads((ANALYSIS / "analysis.json").read_text(encoding="utf-8"))
    means = pd.read_csv(ANALYSIS / "dataset_means.csv")
    suite = json.loads((ROOT / "configs" / "openml_suite.json").read_text(encoding="utf-8"))
    dataset_table(suite)
    comparison_table(analysis)
    secondary_table(analysis)
    sensitivity_table(analysis)
    runtime_table(analysis)
    per_dataset_tables(means)
    for name in ("average_ranks.pdf", "per_dataset_primary_metric.pdf", "runtime.pdf"):
        shutil.copy2(ANALYSIS / name, FIGURES / name)
    for name in ("table_classification.tex", "table_regression.tex", "table_paired.tex",
                 "table_study_classification.tex", "table_study_regression.tex"):
        shutil.copy2(IEEE / "tables" / name, TABLES / f"original_{name}")
    for name in ("paired_differences.pdf", "regression_kumo_intervals.pdf"):
        source = IEEE / "figures" / name
        if source.exists():
            shutil.copy2(source, FIGURES / f"original_{name}")
    for name in ("classification_sample_size_accuracy.png", "regression_sample_size_r2.png",
                 "regression_kumo_intervals.png"):
        shutil.copy2(IEEE / "figures" / name, FIGURES / f"original_{name}")
    for name in ("datasets.tex", "per_dataset_classification.tex", "per_dataset_regression.tex", "original_table_classification.tex",
                 "original_table_regression.tex", "original_table_paired.tex"):
        fit_to_width(TABLES / name)
    print("TMLR tables and figures written to", MANUSCRIPT)


if __name__ == "__main__":
    main()
