import json
import re
import sys
from pathlib import Path

import nbformat
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

PAPER_DIR = Path(__file__).resolve().parent
ROOT = PAPER_DIR.parent
RESULTS = ROOT / "results"
TEX = (PAPER_DIR / "final_submission" / "latex_source" / "main.tex").read_text(encoding="utf-8")
KUMO = "Kumo Tabular"

summary = {t: pd.read_csv(RESULTS / t / "summary.csv").set_index("model") for t in ("classification", "regression")}
by_seed = {t: pd.read_csv(RESULTS / t / "results_by_seed.csv") for t in ("classification", "regression")}
study = {t: pd.read_csv(RESULTS / t / "sample_size_study.csv") for t in ("classification", "regression")}
stats = json.loads((PAPER_DIR / "statistics.json").read_text(encoding="utf-8"))
errors = pd.read_csv(ROOT / "notebooks" / "outputs" / "classification_errors.csv")
reg_nb = nbformat.read(ROOT / "notebooks" / "02_regression_benchmark.ipynb", 4)


def study_value(task, model, n, metric, stat="mean"):
    rows = study[task][(study[task]["model"] == model) & (study[task]["n_train"] == n) & (study[task]["status"] == "ok")]
    return getattr(rows[metric], stat)()


cls, reg = summary["classification"], summary["regression"]
confusion = by_seed["classification"].groupby("model")[["fp", "fn", "tp", "tn"]].sum()
kumo_reg = by_seed["regression"][by_seed["regression"]["model"] == KUMO]
kumo_errors = errors[errors["misclassified_by"].str.contains(KUMO)]
point_output = next(c for c in reg_nb.cells if c.cell_type == "code" and "point_rows" in c.source).outputs[0]["data"]["text/plain"]
corr_output = next(c for c in reg_nb.cells if c.cell_type == "code" and "Correlation of absolute" in c.source)
corr_text = "".join(o.get("text", "") for o in corr_output.outputs)
paired = stats["paired"]
holm_min = min(r["corrected_p_holm"] for r in paired["classification"].values()) if "corrected_p_holm" in next(iter(paired["classification"].values())) else None
diabetes = load_diabetes(as_frame=True, scaled=False)
cancer = load_breast_cancer(as_frame=True)

claims = [
    ("accuracy Kumo 0.989", f"{cls.loc[KUMO, 'accuracy_mean']:.3f}", "0.989"),
    ("accuracy LR 0.977", f"{cls.loc['Logistic Regression', 'accuracy_mean']:.3f}", "0.977"),
    ("recall Kumo 0.976", f"{cls.loc[KUMO, 'recall_mean']:.3f}", "0.976"),
    ("recall LR 0.948", f"{cls.loc['Logistic Regression', 'recall_mean']:.3f}", "0.948"),
    ("R2 Kumo 0.503", f"{reg.loc[KUMO, 'r2_mean']:.3f}", "0.503"),
    ("R2 LR 0.496", f"{reg.loc['Linear Regression', 'r2_mean']:.3f}", "0.496"),
    ("ROC-AUC range low 0.988", f"{cls['roc_auc_mean'].min():.3f}", "0.988"),
    ("ROC-AUC range high 0.998", f"{cls['roc_auc_mean'].max():.3f}", "0.998"),
    ("pooled malignant 210", str(int(confusion.loc[KUMO, ["tp", "fn"]].sum())), "210"),
    ("pooled benign 360", str(int(confusion.loc[KUMO, ["tn", "fp"]].sum())), "360"),
    ("Kumo FN/FP 5/1", f"{confusion.loc[KUMO, 'fn']}/{confusion.loc[KUMO, 'fp']}", "5/1"),
    ("LR FN/FP 11/2", f"{confusion.loc['Logistic Regression', 'fn']}/{confusion.loc['Logistic Regression', 'fp']}", "11/2"),
    ("HGB FN/FP 14/5", f"{confusion.loc['HistGradientBoosting', 'fn']}/{confusion.loc['HistGradientBoosting', 'fp']}", "14/5"),
    ("RF FN/FP 15/11", f"{confusion.loc['Random Forest', 'fn']}/{confusion.loc['Random Forest', 'fp']}", "15/11"),
    ("interval coverage 93.3%", f"{100 * reg.loc[KUMO, 'interval_coverage_mean']:.1f}", "93.3"),
    ("interval coverage sd 3.3%", f"{100 * reg.loc[KUMO, 'interval_coverage_std']:.1f}", "3.3"),
    ("interval width 174.2", f"{reg.loc[KUMO, 'interval_width_mean']:.1f}", "174.2"),
    ("interval width sd 7.8", f"{reg.loc[KUMO, 'interval_width_std']:.1f}", "7.8"),
    ("coverage min 87.6%", f"{100 * kumo_reg['interval_coverage'].min():.1f}", "87.6"),
    ("coverage max 95.5%", f"{100 * kumo_reg['interval_coverage'].max():.1f}", "95.5"),
    ("point estimate mean-of-quantiles row", " ".join(point_output.split("mean of quantiles")[1].split()[:3]), "42.5412 52.3537 0.5133"),
    ("point estimate median row", " ".join(point_output.split("median (q500)")[1].split()[:3]), "42.4779 52.8797 0.5031"),
    ("abs-error corr LR 0.93", re.search(r"Linear Regression error\s+([\d.]+)", corr_text).group(1), "0.93"),
    ("abs-error corr RF 0.76", re.search(r"Random Forest error\s+([\d.]+)", corr_text).group(1), "0.76"),
    ("abs-error corr HGB 0.72", re.search(r"HistGradientBoosting error\s+([\d.]+)", corr_text).group(1), "0.72"),
    ("recall corrected p 0.016", f"{paired['classification']['recall']['corrected_p']:.3f}", "0.016"),
    ("recall corrected CI low 0.009", f"{paired['classification']['recall']['corrected_ci'][0]:.3f}", "0.009"),
    ("recall corrected CI high 0.048", f"{paired['classification']['recall']['corrected_ci'][1]:.3f}", "0.048"),
    ("naive p accuracy < 0.05", str(paired["classification"]["accuracy"]["naive_p"] < 0.05), "True"),
    ("naive p recall < 0.05", str(paired["classification"]["recall"]["naive_p"] < 0.05), "True"),
    ("naive p f1 < 0.05", str(paired["classification"]["f1"]["naive_p"] < 0.05), "True"),
    ("R2 mean advantage 0.007", f"{paired['regression']['r2']['mean_advantage']:.3f}", "0.007"),
    ("R2 advantage sd 0.028", f"{paired['regression']['r2']['sd_advantage']:.3f}", "0.028"),
    ("MAE wins 4", str(paired["regression"]["mae"]["wins"]), "4"),
    ("RMSE wins 3", str(paired["regression"]["rmse"]["wins"]), "3"),
    ("R2 wins 3", str(paired["regression"]["r2"]["wins"]), "3"),
    ("study reg n=25 Kumo 0.429+-0.058", f"{study_value('regression', KUMO, 25, 'r2'):.3f}+-{study_value('regression', KUMO, 25, 'r2', 'std'):.3f}", "0.429+-0.058"),
    ("study reg n=25 LR 0.075+-0.304", f"{study_value('regression', 'Linear Regression', 25, 'r2'):.3f}+-{study_value('regression', 'Linear Regression', 25, 'r2', 'std'):.3f}", "0.075+-0.304"),
    ("study reg n=25 RF 0.388+-0.127", f"{study_value('regression', 'Random Forest', 25, 'r2'):.3f}+-{study_value('regression', 'Random Forest', 25, 'r2', 'std'):.3f}", "0.388+-0.127"),
    ("study cls n=25 LR 0.956", f"{study_value('classification', 'Logistic Regression', 25, 'accuracy'):.3f}", "0.956"),
    ("study cls n=25 Kumo 0.950", f"{study_value('classification', KUMO, 25, 'accuracy'):.3f}", "0.950"),
    ("study cls n=25 HGB acc 0.632", f"{study_value('classification', 'HistGradientBoosting', 25, 'accuracy'):.3f}", "0.632"),
    ("study cls n=25 HGB AUC 0.500", f"{study_value('classification', 'HistGradientBoosting', 25, 'roc_auc'):.3f}", "0.500"),
    ("runtime Kumo cls 1.10", f"{cls.loc[KUMO, 'total_seconds_mean']:.2f}", "1.10"),
    ("runtime Kumo reg 0.63", f"{reg.loc[KUMO, 'total_seconds_mean']:.2f}", "0.63"),
    ("runtime LR cls 0.019", f"{cls.loc['Logistic Regression', 'total_seconds_mean']:.3f}", "0.019"),
    ("runtime LR reg 0.003", f"{reg.loc['Linear Regression', 'total_seconds_mean']:.3f}", "0.003"),
    ("ratio cls ~60", f"{cls.loc[KUMO, 'total_seconds_mean'] / cls.loc['Logistic Regression', 'total_seconds_mean']:.0f}", "59"),
    ("ratio reg ~220", f"{reg.loc[KUMO, 'total_seconds_mean'] / reg.loc['Linear Regression', 'total_seconds_mean']:.0f}", "218"),
    ("Kumo time n=25 cls 0.46", f"{stats['kumo_seconds_by_size']['classification']['25']:.2f}", "0.46"),
    ("Kumo time n=455 cls 1.11", f"{stats['kumo_seconds_by_size']['classification']['455']:.2f}", "1.11"),
    ("Kumo time n=25 reg 0.44", f"{stats['kumo_seconds_by_size']['regression']['25']:.2f}", "0.44"),
    ("Kumo time n=353 reg 0.64", f"{stats['kumo_seconds_by_size']['regression']['353']:.2f}", "0.64"),
    ("error cases >=1 model 38", str(len(errors)), "38"),
    ("error cases all models 4", str(int((errors["models_wrong"] == 4).sum())), "4"),
    ("Kumo errors 6", str(len(kumo_errors)), "6"),
    ("Kumo errors shared by all 4", str(int((kumo_errors["models_wrong"] == 4).sum())), "4"),
    ("of those, malignant with p<0.07: 3", str(int(((kumo_errors["models_wrong"] == 4) & (kumo_errors["actual"] == "malignant") & (kumo_errors["kumo_p_malignant"] < 0.07)).sum())), "3"),
    ("Kumo error shared only with RF: 1", str(int((kumo_errors["misclassified_by"] == "Kumo Tabular, Random Forest").sum())), "1"),
    ("Kumo-only error: 1", str(int((kumo_errors["misclassified_by"] == KUMO).sum())), "1"),
    ("malignant count 212", str(int((cancer.target == 0).sum())), "212"),
    ("benign count 357", str(int((cancer.target == 1).sum())), "357"),
    ("diabetes target min 25", f"{diabetes.target.min():.0f}", "25"),
    ("diabetes target max 346", f"{diabetes.target.max():.0f}", "346"),
    ("train/test cls 455/114", f"{by_seed['classification']['n_train'].iloc[0]}/{by_seed['classification']['n_test'].iloc[0]}", "455/114"),
    ("train/test reg 353/89", f"{by_seed['regression']['n_train'].iloc[0]}/{by_seed['regression']['n_test'].iloc[0]}", "353/89"),
]
rerun = {t: pd.read_csv(PAPER_DIR / "final_submission" / "reproduction" / t / "summary.csv").set_index("model") for t in ("classification", "regression")}
claims += [
    ("re-run Kumo cls 1.22", f"{rerun['classification'].loc[KUMO, 'total_seconds_mean']:.2f}", "1.22"),
    ("re-run LR cls 0.052", f"{rerun['classification'].loc['Logistic Regression', 'total_seconds_mean']:.3f}", "0.052"),
    ("re-run Kumo reg 0.68", f"{rerun['regression'].loc[KUMO, 'total_seconds_mean']:.2f}", "0.68"),
    ("re-run LR reg 0.023", f"{rerun['regression'].loc['Linear Regression', 'total_seconds_mean']:.3f}", "0.023"),
    ("re-run ratio cls ~23", f"{rerun['classification'].loc[KUMO, 'total_seconds_mean'] / rerun['classification'].loc['Logistic Regression', 'total_seconds_mean']:.0f}", "23"),
    ("re-run ratio reg ~30", f"{rerun['regression'].loc[KUMO, 'total_seconds_mean'] / rerun['regression'].loc['Linear Regression', 'total_seconds_mean']:.0f}", "30"),
]
if holm_min is not None:
    claims.append(("smallest Holm-adjusted p 0.081", f"{holm_min:.3f}", "0.081"))

failures = 0
for name, actual, expected in claims:
    ok = actual == expected
    failures += not ok
    print(f"{'OK  ' if ok else 'FAIL'} {name:45s} source={actual}")
print(f"\n{len(claims) - failures}/{len(claims)} manuscript numbers match their source files")
for literal in ["0.989", "0.977", "0.976", "0.948", "0.503", "0.496", "93.3", "174.2", "0.016", "0.081", "0.429", "27.5M", "28.5M"]:
    if literal not in TEX:
        print("not found in manuscript text:", literal)
sys.exit(1 if failures else 0)
