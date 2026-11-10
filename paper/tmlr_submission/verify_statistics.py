import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, rankdata

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "openml_benchmark"
KUMO = "Kumo Tabular"
METRICS = {"classification": {"log_loss": False, "accuracy": True, "roc_auc": True, "macro_f1": True},
           "regression": {"rmse": False, "mae": False, "r2": True}}


def friedman(table: np.ndarray) -> tuple[float, float]:
    n, k = table.shape
    ranks = np.vstack([rankdata(row) for row in table])
    rank_sums = ranks.sum(axis=0)
    statistic = 12 / (n * k * (k + 1)) * np.sum(rank_sums ** 2) - 3 * n * (k + 1)
    ties = sum(np.sum(c ** 3 - c) for row in table for c in [np.unique(row, return_counts=True)[1]])
    statistic /= 1 - ties / (n * (k ** 3 - k))
    return statistic, chi2.sf(statistic, k - 1)


def exact_wilcoxon(differences: np.ndarray) -> float:
    d = differences[np.abs(differences) > 1e-12]
    ranks = rankdata(np.abs(d))
    observed = np.sum(ranks[d > 0])
    total = ranks.sum()
    extreme = max(observed, total - observed)
    count = 0
    for signs in itertools.product((0, 1), repeat=len(d)):
        w = np.dot(signs, ranks)
        if w >= extreme - 1e-9 or w <= total - extreme + 1e-9:
            count += 1
    return min(1.0, count / 2 ** len(d))


def holm(p: dict) -> dict:
    keys = sorted(p, key=p.get)
    out, running = {}, 0.0
    for i, key in enumerate(keys):
        running = max(running, min(1.0, (len(keys) - i) * p[key]))
        out[key] = running
    return out


def main() -> None:
    raw = pd.concat([pd.read_csv(p) for p in sorted((RESULTS / "raw").glob("*.csv"))], ignore_index=True)
    reported = json.loads((RESULTS / "analysis" / "analysis.json").read_text(encoding="utf-8"))
    mismatches, checks = [], 0
    for task, metrics in METRICS.items():
        rows = raw[raw["task"] == task]
        for metric, higher in metrics.items():
            values = {}
            for record in rows[["data_id", "model", metric]].itertuples(index=False):
                values.setdefault((record.data_id, record.model), []).append(getattr(record, metric))
            datasets = sorted({k[0] for k in values})
            models = sorted({k[1] for k in values})
            table = np.array([[math.fsum(values[(d, m)]) / len(values[(d, m)]) for m in models] for d in datasets])
            oriented = -table if higher else table
            ranks = np.vstack([rankdata(row) for row in oriented]).mean(axis=0)
            stat, p_friedman = friedman(oriented)
            block = reported[task]["all"][metric]
            checks += 2
            if abs(p_friedman - block["friedman_p"]) > 1e-9 * max(1, p_friedman) + 1e-15:
                mismatches.append(f"{task}/{metric} Friedman p {p_friedman} vs {block['friedman_p']}")
            if abs(stat - block["friedman_statistic"]) > 1e-6:
                mismatches.append(f"{task}/{metric} Friedman statistic {stat} vs {block['friedman_statistic']}")
            for model, rank in zip(models, ranks):
                checks += 1
                if abs(round(rank, 3) - block["average_rank"][model]) > 1e-9:
                    mismatches.append(f"{task}/{metric} rank {model} {rank} vs {block['average_rank'][model]}")
            kumo = models.index(KUMO)
            p_values, wtl = {}, {}
            for j, model in enumerate(models):
                if model == KUMO:
                    continue
                advantage = (table[:, kumo] - table[:, j]) if higher else (table[:, j] - table[:, kumo])
                p_values[model] = exact_wilcoxon(advantage)
                wtl[model] = (int((advantage > 1e-12).sum()), int((np.abs(advantage) <= 1e-12).sum()),
                              int((advantage < -1e-12).sum()))
            adjusted = holm(p_values)
            for row in block["kumo_vs_baselines"]:
                model = row["baseline"]
                checks += 3
                if abs(p_values[model] - row["wilcoxon_p"]) > 1e-9:
                    mismatches.append(f"{task}/{metric} Wilcoxon {model} {p_values[model]} vs {row['wilcoxon_p']}")
                if abs(adjusted[model] - row["holm_p"]) > 1e-9:
                    mismatches.append(f"{task}/{metric} Holm {model} {adjusted[model]} vs {row['holm_p']}")
                if wtl[model] != (row["kumo_wins"], row["ties"], row["kumo_losses"]):
                    mismatches.append(f"{task}/{metric} W/T/L {model} {wtl[model]} vs reported")
    print(f"{checks} quantities recomputed independently; mismatches: {len(mismatches)}")
    for line in mismatches:
        print("  ", line)


if __name__ == "__main__":
    main()
