# Experimental Gap Analysis

Date: 9 November 2026. Hardware: NVIDIA GeForce GTX 980M (8 GB, compute capability 5.2), Intel Core i7-6700HQ, 16 GB RAM; disk space available: ~386 GB on F:. No paid services were used.

## Assessment of the original evidence

The original study supports only dataset-specific statements about two small tables. TMLR evaluates whether claims are supported by evidence and whether the work would interest its audience. A benchmark comparison of a new tabular foundation model can meet that bar if its claims are scoped to what a fixed, external set of datasets supports. The decisive gap was breadth (Critical), followed by baseline strength and inference design (Major).

## Experiments considered

| Candidate | Scientific purpose | Decision | Reason |
|---|---|---|---|
| **OpenML suite (CC18 + CTR23 subset)** | Test whether the two-dataset findings generalize across an external, curated set of small numerical tables | **Executed** | Public data, official splits, fits on the available GPU; pre-specified filters avoid cherry-picking |
| **XGBoost, LightGBM, CatBoost** | Compare against the gradient-boosting libraries that define the state of practice | **Executed** (defaults) | Free, pip-installable, CPU; installation approved by the author |
| **Extra Trees** | Second randomized tree ensemble, part of scikit-learn | **Executed** (500 trees) | No new dependency |
| TabPFN v2 | Closest comparable tabular foundation model | Not run | Weights require accepting a licence through a Hugging Face account; no account may be created on the author's behalf, and installing it was not approved |
| TabICL | Related foundation model | Not run | Not approved for installation in this revision |
| Kumo `medium` / `large` | Effect of checkpoint size | Not run | Not requested; GPU memory (8 GB, no float16 autocast on compute capability 5.2) makes the 215M-parameter model costly. Stated as a limitation |
| Hyperparameter tuning of baselines | Upper bound for classical models | Not run | Changes the research question (out-of-the-box comparison) and multiplies compute; framed as a limitation |
| Datasets > 2,000 rows (subsampled) or with categorical/missing values | Breadth beyond the small numerical regime | Not run | Would require subsampling and encoding choices that add confounds; excluded by the pre-specified filters |
| Repeated 10×10 cross-validation | Tighter within-dataset estimates | Not run | Cross-dataset inference uses dataset means; the first repetition of the official splits is used uniformly |
| Ablation of Kumo's ensemble size (estimators) | Sensitivity to inference settings | Not run | Out of scope for this revision; noted as future work |

## Protocol of the executed experiment

| Item | Specification |
|---|---|
| Datasets | 15 classification + 7 regression selected from OpenML-CC18 / OpenML-CTR23 (15 + 6 evaluated; `forest_fires` excluded at load because of undeclared text columns); 150–2,000 rows, ≤ 60 features, numerical only, no missing values, ≤ 10 classes |
| Splits | Official OpenML task splits, 10-fold CV, repetition 0 |
| Seeds | Fold index (0–9) for all models and Kumo's generator |
| Models | Kumo Tabular `small` (8 estimators); logistic/linear regression; random forest; extra trees; histogram gradient boosting; XGBoost; LightGBM; CatBoost |
| Metrics | Primary: log loss, RMSE. Secondary: accuracy, macro-F1, one-vs-rest macro ROC-AUC; MAE, R² |
| Fairness | Identical folds and seeds; no tuning for any model; preprocessing inside pipelines (standardization for logistic regression only) |
| Raw evidence | `results/openml_benchmark/raw/*.csv` (one file per dataset, one row per fold and model), `results_by_fold.csv`, `run_log.txt`, `console_output.txt`, `environment.json` |
| Analysis | `scripts/analyze_openml_benchmark.py` → `results/openml_benchmark/analysis/` |

## Execution record

Execution status, failures and timings are documented in `results/openml_benchmark/run_log.txt`. The verification of the outputs (row counts, failures, consistency with the original study) is reported in `paper/tmlr_submission/REPRODUCIBILITY.md`.
