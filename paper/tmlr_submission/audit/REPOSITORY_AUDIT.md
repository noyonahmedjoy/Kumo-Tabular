# Repository Audit

Repository: `F:\project\Kumo Tabular` (public mirror: the author's GitHub repository; not named in the anonymized submission). Audit date: 9 November 2026. Starting state: `main` at merge commit `2dbca1a`, working tree clean.

Every statement below was checked against the code, configuration, result files or git history, not against README descriptions.

## 1. Architecture

| Path | Content | Lines / files |
|---|---|---|
| `src/data_loader.py` | Loads scikit-learn breast cancer (malignant recoded as 1) and diabetes (`scaled=False`); `split_dataset` (80/20, stratified for classification); `validate_split` (index disjointness, alignment, no target column); `subsample_training` | |
| `src/kumo_model.py` | `KumoTabularPredictor` around `sdm.models.KumoTabular`: device resolution, float16 autocast only when compute capability ≥ 7.0, seeded `torch.Generator`, class-order alignment by column name, quantile output validation, warm-up | |
| `src/baseline_models.py` | Logistic regression (standardised, `max_iter=5000`), random forest (500 trees), histogram gradient boosting; linear regression for regression | |
| `src/evaluation.py` | Binary classification metrics (positive class 1), regression metrics, interval coverage, prediction validation | |
| `src/benchmarking.py` | Runs Kumo and baselines on shared splits, timing, aggregation, leaderboard, paired comparison, training-size study, `run_experiment` | |
| `src/visualization.py` | All original figures | |
| `scripts/` | `run_all.py`, `run_classification.py`, `run_regression.py`, `build_notebooks.py` | |
| `tests/` | 37 tests (data loading, leakage, metrics, benchmarking, real-model tests) | |
| `notebooks/` | Three generated notebooks, also published on Kaggle | 3 |
| `results/` | Per-split results, summaries, training-size study, figures for the two original experiments | 24 files |
| `paper/` | IEEE Access package, asset and number-check scripts, statistics | |

Source, scripts and tests total 2,038 lines. Dependencies are pinned in `requirements.txt` only for `structured-data-models==0.2.1`; other packages are unpinned.

## 2. Research objective and claimed contributions (current IEEE manuscript)

Objective: compare the pretrained in-context learner Kumo Tabular (`small` checkpoint) with untuned classical models on small tabular data.

Claimed contributions: (i) a reproducible comparison on identical splits, including a training-size study and runtimes; (ii) a paired analysis with a split-overlap correction and multiple-comparison adjustment; (iii) an analysis of where the pretrained model helps and where it does not. The contribution is empirical: no new method is proposed.

## 3. Dataset inventory

| Dataset | Source | Rows | Features | Target | Used in |
|---|---|---|---|---|---|
| Breast Cancer Wisconsin (Diagnostic) | `sklearn.datasets.load_breast_cancer` | 569 | 30 numerical | malignant 212 / benign 357 | classification experiment |
| Diabetes | `sklearn.datasets.load_diabetes(scaled=False)` | 442 | 10 numerical | progression, 25–346 | regression experiment |

No data files are stored in the repository. Both datasets are bundled with scikit-learn.

## 4. Experimental protocol (as implemented)

- Five 80/20 splits with seeds 42–46, stratified for classification; identical splits for all models (`split_dataset`).
- Kumo: `small`, 8 estimators, estimator batch size 1, seeded generator; training rows as context and test features as query (target column rejected for queries). The median quantile `q500` is the regression point prediction; `q050`–`q950` gives an interval.
- Baselines: scikit-learn defaults with fixed `random_state`; no tuning for any model.
- Timing: fit plus predict per split; one Kumo warm-up call (on training rows) excluded; Kumo on GPU, baselines on CPU.
- Training-size study: seeds 42–44; training subsamples 25/50/100/200 (+400) and the full set; full test split.

## 5. Statistical methodology (as implemented)

Per metric: paired difference between Kumo and the strongest baseline on each of the five splits; naive paired t-test; Nadeau–Bengio corrected resampled t-test (test/train ratio ≈ 0.25) with a 95% interval; Holm adjustment within each task; exact Wilcoxon signed-rank test; win/tie/loss counts (`paper/build_paper_assets.py`). Assessed in `STATISTICAL_AUDIT.md`.

## 6. Reproducibility status

| Evidence | Status |
|---|---|
| Re-run in the original environment (1 Nov 2026) | 172/172 result rows identical |
| Kaggle notebooks (T4, PyTorch 2.11, float16) | Kumo, logistic/linear regression and random forest match at reported precision; histogram gradient boosting differs |
| Tables regenerated from `results/` | identical |
| Unit tests | 37/37 pass (9 Nov 2026) |

## 7. Missing evidence and contradictions

| Item | Finding |
|---|---|
| Breadth of evidence | Only two datasets. Every cross-method claim rests on two tables of 569 and 442 rows. For TMLR this is the main weakness: no claim about "small tabular data" in general is supported. |
| Baseline coverage | No XGBoost, LightGBM, CatBoost or Extra Trees, although the paper's related work presents gradient-boosted trees as the reference standard. |
| Unit of statistical inference | Splits of one dataset are the unit; generalization across datasets cannot be assessed. |
| Multiclass support | `classification_metrics` assumes binary labels `[0, 1]` and baselines use `predict_proba[:, 1]`; the code cannot evaluate multiclass tasks. |
| Runtime comparison | GPU (Kumo) versus CPU (baselines); timings of a few milliseconds varied up to eightfold between runs. Runtime ratios are hardware-dependent and only descriptive. |
| Environment sensitivity | Histogram gradient boosting results differ between scikit-learn installations (local vs Kaggle). Dependencies other than `structured-data-models` are unpinned. |
| Single checkpoint size | Only `small` was evaluated; the developers' headline claims concern larger checkpoints. |
| Manuscript vs code | No contradiction found in the current IEEE manuscript (73/73 quoted numbers verified on 2 Nov 2026). |
| Authorship provenance | Code, experiments, statistical analysis and manuscript text were produced by an AI system (Claude) working from author specifications; ChatGPT assisted planning, prompt preparation and review (author statement). The TMLR FAQ expects the "ideas, claims and results" to be human-sourced; see `AI_ASSISTANCE_DISCLOSURE.md`. |

## 8. Critical weaknesses and corrective actions

| # | Weakness | Severity | Corrective action in this revision |
|---|---|---|---|
| 1 | Two datasets only | Critical | Add a pre-registered suite of 22 OpenML datasets (OpenML-CC18 and OpenML-CTR23, fixed inclusion filters) with official OpenML splits |
| 2 | Weak baseline set | Major | Add Extra Trees, XGBoost, LightGBM and CatBoost (defaults) |
| 3 | Inference across datasets impossible | Major | Use datasets as the unit: Friedman test, pairwise Wilcoxon signed-rank tests with Holm correction, average ranks, win/tie/loss counts |
| 4 | Binary-only metrics | Major (blocks multiclass) | Add multiclass-capable metrics (accuracy, macro-F1, one-vs-rest ROC-AUC, log loss) without changing the original binary functions |
| 5 | Runtime confound | Moderate | Report runtimes per hardware, descriptively, and drop ratio claims from the abstract |
| 6 | Unpinned dependencies | Moderate | Record exact versions in the new results' environment file |
| 7 | Single checkpoint | Moderate | Stated as a limitation (no larger-checkpoint runs on this 8 GB GPU in this revision) |
| 8 | AI provenance vs TMLR expectations | High (editorial) | Truthful first-page disclosure; author must review and own every claim before submission |
