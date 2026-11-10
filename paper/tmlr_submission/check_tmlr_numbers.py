import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "results" / "openml_benchmark"
TEX = "\n".join(p.read_text(encoding="utf-8") for p in sorted((Path(__file__).parent / "manuscript").rglob("*.tex")))
KUMO = "Kumo Tabular"

analysis = json.loads((BENCH / "analysis" / "analysis.json").read_text(encoding="utf-8"))
means = pd.read_csv(BENCH / "analysis" / "dataset_means.csv")
raw = pd.read_csv(BENCH / "results_by_fold.csv")
cls, reg = analysis["classification"], analysis["regression"]


def vs(task, metric, baseline, field, block="all"):
    return next(r[field] for r in analysis[task][block][metric]["kumo_vs_baselines"] if r["baseline"] == baseline)


def relative_to_best(task, metric, lower_is_better=True):
    table = means[means["task_type"] == task].pivot(index="dataset", columns="model", values=metric)
    if lower_is_better:
        best = table.drop(columns=KUMO).min(axis=1)
        return (best - table[KUMO]) / best
    return table[KUMO] - table.drop(columns=KUMO).max(axis=1)


rel_logloss = relative_to_best("classification", "log_loss")
rel_rmse = relative_to_best("regression", "rmse")
acc_diff = relative_to_best("classification", "accuracy", lower_is_better=False) * 100
clim = means[(means["dataset"] == "climate-model-simulation-crashes")].set_index("model")["log_loss"]
rmse_table = means[means["task_type"] == "regression"].pivot(index="dataset", columns="model", values="rmse")
zernike = raw[raw["dataset"] == "mfeat-zernike"].groupby("model")["total_seconds"].sum()
cr, rr = cls["runtime_seconds_per_fold"]["median"], reg["runtime_seconds_per_fold"]["median"]
baselines_cls = [r["baseline"] for r in cls["all"]["log_loss"]["kumo_vs_baselines"]]

claims = [
    ("model runs 1,680", str(len(raw)), "1680"),
    ("datasets 21 (15 + 6)", f"{cls['all']['log_loss']['datasets']}+{reg['all']['rmse']['datasets']}", "15+6"),
    ("cls log-loss rank 1.07", f"{cls['all']['log_loss']['average_rank'][KUMO]:.2f}", "1.07"),
    ("cls Friedman p < 0.001", str(cls["all"]["log_loss"]["friedman_p"] < 0.001), "True"),
    ("15/15 vs six baselines", str(all(vs("classification", "log_loss", b, "kumo_wins") == 15
                                       for b in baselines_cls if b != "Logistic Regression")), "True"),
    ("14/15 vs logistic regression", str(vs("classification", "log_loss", "Logistic Regression", "kumo_wins")), "14"),
    ("all Holm p < 0.001", str(all(r["holm_p"] < 0.001 for r in cls["all"]["log_loss"]["kumo_vs_baselines"])), "True"),
    ("12.8% vs CatBoost", f"{100 * vs('classification', 'log_loss', 'CatBoost', 'median_relative_advantage'):.1f}", "12.8"),
    ("54.5% vs extra trees", f"{100 * vs('classification', 'log_loss', 'Extra Trees', 'median_relative_advantage'):.1f}", "54.5"),
    ("better than per-dataset best on 14/15", str(int((rel_logloss > 0).sum())), "14"),
    ("climate: LR 0.1311", f"{clim['Logistic Regression']:.4f}", "0.1311"),
    ("climate: Kumo 0.1313", f"{clim[KUMO]:.4f}", "0.1313"),
    ("ROC-AUC rank 1.33", f"{cls['all']['roc_auc']['average_rank'][KUMO]:.2f}", "1.33"),
    ("accuracy rank 1.77", f"{cls['all']['accuracy']['average_rank'][KUMO]:.2f}", "1.77"),
    ("macro-F1 rank 2.20", f"{cls['all']['macro_f1']['average_rank'][KUMO]:.2f}", "2.20"),
    ("secondary wins >= 11", str(min(r["kumo_wins"] for m in ("accuracy", "roc_auc", "macro_f1")
                                     for r in cls["all"][m]["kumo_vs_baselines"])), "11"),
    ("accuracy diabetes -1.8", f"{acc_diff['diabetes']:.1f}", "-1.8"),
    ("accuracy climate -1.1", f"{acc_diff['climate-model-simulation-crashes']:.1f}", "-1.1"),
    ("accuracy kc2 -0.2", f"{acc_diff['kc2']:.1f}", "-0.2"),
    ("accuracy pc3 equal", f"{acc_diff['pc3']:.1f}", "0.0"),
    ("accuracy vehicle +11.5", f"{acc_diff['vehicle']:.1f}", "11.5"),
    ("accuracy balance-scale +10.4", f"{acc_diff['balance-scale']:.1f}", "10.4"),
    ("below best accuracy on 3 datasets", str(int((acc_diff < -1e-9).sum())), "3"),
    ("reg RMSE rank 1.67", f"{reg['all']['rmse']['average_rank'][KUMO]:.2f}", "1.67"),
    ("reg CatBoost rank 2.67", f"{reg['all']['rmse']['average_rank']['CatBoost']:.2f}", "2.67"),
    ("reg extra trees rank 3.50", f"{reg['all']['rmse']['average_rank']['Extra Trees']:.2f}", "3.50"),
    ("reg Friedman p < 0.001", str(reg["all"]["rmse"]["friedman_p"] < 0.001), "True"),
    ("least squares last everywhere", str((rmse_table.rank(axis=1)["Linear Regression"] == 8).all()), "True"),
    ("reg smallest Holm p 0.22", f"{min(r['holm_p'] for r in reg['all']['rmse']['kumo_vs_baselines']):.2f}", "0.22"),
    ("reg better than best baseline on 4/6", str(int((rel_rmse > 0).sum())), "4"),
    ("airfoil 44% lower", f"{100 * rel_rmse['airfoil_self_noise']:.0f}", "44"),
    ("energy 44% higher than XGBoost", f"{100 * (rmse_table.loc['energy_efficiency', KUMO] / rmse_table.loc['energy_efficiency', 'XGBoost'] - 1):.0f}", "44"),
    ("energy best baseline XGBoost", rmse_table.drop(columns=KUMO).loc["energy_efficiency"].idxmin(), "XGBoost"),
    ("red_wine 5% higher than extra trees", f"{100 * (rmse_table.loc['red_wine', KUMO] / rmse_table.loc['red_wine', 'Extra Trees'] - 1):.0f}", "5"),
    ("red_wine best baseline extra trees", rmse_table.drop(columns=KUMO).loc["red_wine"].idxmin(), "Extra Trees"),
    ("reg MAE rank 1.33", f"{reg['all']['mae']['average_rank'][KUMO]:.2f}", "1.33"),
    ("reg R2 rank 1.67", f"{reg['all']['r2']['average_rank'][KUMO]:.2f}", "1.67"),
    ("without wdbc rank 1.07", f"{cls['excluding_overlap_with_original_study']['log_loss']['average_rank'][KUMO]:.2f}", "1.07"),
    ("without wdbc all Holm < 0.05", str(all(r["holm_p"] < 0.05 for r in cls["excluding_overlap_with_original_study"]["log_loss"]["kumo_vs_baselines"])), "True"),
    ("families: 11 units", str(cls["dataset_families_collapsed"]["log_loss"]["datasets"]), "11"),
    ("families: rank 1.09", f"{cls['dataset_families_collapsed']['log_loss']['average_rank'][KUMO]:.2f}", "1.09"),
    ("families: adjusted p 0.007", f"{max(r['holm_p'] for r in cls['dataset_families_collapsed']['log_loss']['kumo_vs_baselines']):.3f}", "0.007"),
    ("runtime Kumo cls 1.27 s", f"{cr[KUMO]:.2f}", "1.27"),
    ("runtime Kumo reg 1.14 s", f"{rr[KUMO]:.2f}", "1.14"),
    ("ratio XGBoost cls ~7", f"{cr[KUMO] / cr['XGBoost']:.0f}", "7"),
    ("ratio LightGBM cls ~7-8", str(7 <= cr[KUMO] / cr["LightGBM"] < 8), "True"),
    ("ratio reg 9 to 11", f"{rr[KUMO] / rr['XGBoost']:.0f}-{rr[KUMO] / rr['LightGBM']:.0f}", "9-11"),
    ("ratio logistic ~48", f"{cr[KUMO] / cr['Logistic Regression']:.0f}", "48"),
    ("ratio least squares ~300", str(290 <= rr[KUMO] / rr["Linear Regression"] <= 320), "True"),
    ("zernike HistGB 1,113 s", f"{zernike['HistGradientBoosting']:.0f}", "1113"),
    ("zernike CatBoost 841 s", f"{zernike['CatBoost']:.0f}", "841"),
    ("zernike Kumo 650 s", f"{zernike[KUMO]:.0f}", "650"),
    ("zernike others <= 21 s", str(max(zernike[m] for m in ("XGBoost", "LightGBM", "Extra Trees", "Logistic Regression")) <= 21.5), "True"),
]
failures = 0
for name, actual, expected in claims:
    ok = actual == expected
    failures += not ok
    print(f"{'OK  ' if ok else 'FAIL'} {name:<42} source={actual}")
print(f"\n{len(claims) - failures}/{len(claims)} manuscript numbers match the result files")
for literal in ["1.07", "0.1311", "0.1313", "1{,}113", "12.8", "54.5"]:
    print(f"   literal {literal!r} present in manuscript: {literal in TEX}")
sys.exit(1 if failures else 0)
