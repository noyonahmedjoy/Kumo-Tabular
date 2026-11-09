# Kaggle execution evidence

Recorded on 2 November 2026 by opening the three public notebooks in a web browser. Values are transcribed from the run information on each notebook page and from the stored outputs in the rendered notebook (`__results__.html`). Nothing was re-run on Kaggle during this check.

## Run information

| Notebook | URL | Version | Run status | Run time | Accelerator |
|---|---|---|---|---|---|
| Classification benchmark | https://www.kaggle.com/code/nayonahmedlol/nvidia-kumo-tabular-classification-benchmark | 1 of 1 | "65.7 second run - successful" | 1m 6s | GPU T4 x2 |
| Regression benchmark | https://www.kaggle.com/code/nayonahmedlol/nvidia-kumo-tabular-regression-benchmark | 1 of 1 | "92.9 second run - successful" | 1m 33s | GPU T4 x2 |
| Experimental analysis | https://www.kaggle.com/code/nayonahmedlol/kumo-tabular-vs-traditional-ml-experiment | 1 of 1 | "126.7 second run - successful" | 2m 7s | GPU T4 x2 |

Environment printed by all three notebooks: Python 3.13.15, PyTorch 2.11.0+cu128 (CUDA 12.8), Tesla T4 (14.6 GB), Hugging Face reachable, structured-data-models 0.2.1 installed, "Loaded Kumo Tabular (small) on cuda, autocast=on". The scikit-learn version is not printed.

## Classification (five splits, mean ± standard deviation)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Time (s) |
|---|---|---|---|---|---|---|
| Kumo Tabular | 0.9895 ± 0.0096 | 0.9953 ± 0.0104 | 0.9762 ± 0.0292 | 0.9854 ± 0.0136 | 0.9983 ± 0.0028 | 0.4119 |
| Logistic Regression | 0.9772 ± 0.0118 | 0.9902 ± 0.0134 | 0.9476 ± 0.0310 | 0.9682 ± 0.0168 | 0.9952 ± 0.0041 | 0.0146 |
| Random Forest | 0.9544 ± 0.0130 | 0.9475 ± 0.0308 | 0.9286 ± 0.0238 | 0.9376 ± 0.0178 | 0.9883 ± 0.0112 | 1.2935 |
| HistGradientBoosting | 0.9596 ± 0.0253 | 0.9749 ± 0.0358 | 0.9143 ± 0.0433 | 0.9433 ± 0.0356 | 0.9928 ± 0.0066 | 0.2298 |

Paired comparison with logistic regression (wins/ties/losses, mean advantage): accuracy 5/0/0, +0.0123; precision 2/3/0, +0.0051; recall 5/0/0, +0.0286; F1 5/0/0, +0.0172; ROC-AUC 4/1/0, +0.0030.

Pooled false positives / false negatives: Kumo Tabular 1 / 5, Logistic Regression 2 / 11, Random Forest 11 / 15, HistGradientBoosting 5 / 18.

## Regression (five splits, mean ± standard deviation)

| Model | MAE | RMSE | R² | Time (s) |
|---|---|---|---|---|
| Kumo Tabular | 42.4790 ± 3.0952 | 52.8802 ± 3.4689 | 0.5031 ± 0.0392 | 0.5156 |
| Linear Regression | 43.7464 ± 3.5858 | 53.2724 ± 4.0281 | 0.4959 ± 0.0458 | 0.0106 |
| Random Forest | 45.2268 ± 2.6254 | 56.1636 ± 3.3113 | 0.4403 ± 0.0218 | 1.2421 |
| HistGradientBoosting | 46.2658 ± 1.1605 | 58.7832 ± 1.4546 | 0.3847 ± 0.0413 | 0.1691 |

Paired comparison with linear regression: MAE 4/0/1, +1.2674; RMSE 3/0/2, +0.3922; R² 3/0/2, +0.0072.
Kumo 90% interval: coverage 0.9326 ± 0.0328, mean width 174.2425.
Point estimates: median (q500) MAE 42.4790, RMSE 52.8802, R² 0.5031; mean of quantiles MAE 42.5412, RMSE 52.3534, R² 0.5133.
Correlation of absolute errors with Kumo (seed 42): Linear Regression 0.93, Random Forest 0.76, HistGradientBoosting 0.74.

## Experimental analysis notebook

Completed runs: classification 20 of 20, regression 20 of 20. Its summary tables, paired comparison and training-size tables match the two notebooks above and the manuscript for Kumo Tabular, logistic/linear regression and random forests. Per-split times in this notebook: classification Kumo 0.428 s vs logistic regression 0.015 s; regression Kumo 0.419 s vs linear regression 0.071 s.

Training-size study (mean over seeds 42–44):

| Training rows | Kumo acc. | LR acc. | RF acc. | HGB acc. | Kumo R² | LinReg R² | RF R² | HGB R² |
|---|---|---|---|---|---|---|---|---|
| 25 | 0.950 | 0.956 | 0.930 | 0.632 | 0.429 | 0.075 | 0.388 | −0.097 |
| 50 | 0.953 | 0.953 | 0.933 | 0.909 | 0.437 | 0.331 | 0.418 | 0.348 |
| 100 | 0.956 | 0.956 | 0.939 | 0.944 | 0.486 | 0.467 | 0.433 | 0.410 |
| 200 | 0.985 | 0.971 | 0.947 | 0.947 | 0.505 | 0.506 | 0.438 | 0.391 |
| 400 | 0.982 | 0.974 | 0.953 | 0.959 | – | – | – | – |
| full (455 / 353) | 0.988 | 0.974 | 0.959 | 0.947 | 0.526 | 0.507 | 0.440 | 0.381 |

## Comparison with the manuscript

| Result | Agreement |
|---|---|
| Kumo Tabular, all metrics | Match at the precision reported in the manuscript. Unrounded values differ in the fourth decimal for ROC-AUC (0.9983 vs 0.9982) and the third or fourth decimal for regression, consistent with float16 autocast on the T4 versus float32 locally. |
| Logistic/linear regression and random forests | Identical. |
| HistGradientBoosting | Differs (e.g., accuracy 0.9596 vs 0.9667, false negatives 18 vs 14, regression MAE 46.27 vs 45.85, full-size R² 0.381 vs 0.406, error correlation 0.74 vs 0.72). The cause has not been verified; a different scikit-learn version on Kaggle is the likely explanation. No conclusion of the manuscript depends on HistGradientBoosting's exact values. |
| Paired comparisons and statistical conclusions | Identical win/tie/loss counts and mean advantages (to rounding). |
| Runtimes | Different hardware; Kumo 0.41–0.52 s per split on the T4 versus 0.63–1.10 s on the GTX 980M. |

## How to re-check manually

1. Open each URL above while signed out of Kaggle; the run status is shown under "Logs" and "Version 1 of 1" next to the version history.
2. Scroll the "Notebook" tab to the "Run the benchmark" and "Results" sections and compare the tables with this file.
3. Open the "Output" tab to download the CSV files the notebooks wrote (`outputs/*.csv`).
