# Reproducibility Report

Last updated: 2 November 2026.

## Summary

| Evidence | Scope | Result |
|---|---|---|
| Re-run of all experiments in the original environment (GTX 980M, Windows, PyTorch 2.14.1), 1 Nov 2026 | All 172 per-split and training-size result rows | Identical to `results/` (maximum difference 0) |
| Author-run Kaggle notebooks in a different environment (Tesla T4, Python 3.13.15, PyTorch 2.11.0, float16 autocast), recorded 2 Nov 2026 | Classification, regression and analysis notebooks | Kumo Tabular, logistic/linear regression and random forest results match the manuscript at its reported precision; HistGradientBoosting differs slightly |
| Regeneration of the manuscript tables and statistics from `results/`, 2 Nov 2026 | Tables 4–10, Table 8 statistics, Fig. 7 | Byte-identical to the tables in the source package |
| Check of numbers quoted in the text (`paper/check_manuscript_numbers.py`), 2 Nov 2026 | 73 numerical claims | 73 of 73 match their source files |
| Unit tests | 37 tests, including two that load the real model | All pass |

**Independence.** No third party has reproduced the results. The local re-run was done by the same AI system on the same machine, which shows determinism but not independence. The Kaggle runs used the same code but a different platform, GPU, Python and PyTorch version, and reduced-precision arithmetic, and were executed under the author's account. They are the strongest available evidence that the results do not depend on the original machine. They are still not an independent reproduction by someone else.

## Environments

| | Original / local re-run | Kaggle |
|---|---|---|
| Hardware | NVIDIA GeForce GTX 980M (8 GB), Intel Core i7-6700HQ | Tesla T4 (14.6 GB), "GPU T4 x2" |
| OS / Python | Windows 10, Python 3.12.16 | Kaggle Linux image, Python 3.13.15 |
| PyTorch | 2.14.1+cu126 | 2.11.0+cu128 |
| structured-data-models | 0.2.1 | 0.2.1 |
| scikit-learn | 1.9.1 | not printed |
| Kumo precision | float32 (autocast off, compute capability 5.2) | float16 autocast on |

## 1. Methodology vs. code and configuration

| Manuscript statement | Artifact | Consistent |
|---|---|---|
| Five splits, seeds 42–46, 80/20, stratified for classification | `configs/experiment_config.yaml` (`seeds`, `test_size: 0.2`); `split_dataset` stratifies classification | Yes |
| Kumo `small`, 8 estimators, estimator batch size 1, seeded generator | `kumo` section of the config; `torch.Generator(...).manual_seed(seed)` in `src/kumo_model.py` | Yes |
| float16 autocast only on GPUs with compute capability ≥ 7.0 | `autocast: auto`, `autocast_supported()`; Kaggle T4 printed "autocast=on" | Yes |
| Median `q500` point prediction, `q050`–`q950` interval | `regression_point_quantile`, `regression_interval` | Yes |
| Logistic regression standardised, `max_iter=5000`; random forest 500 trees; defaults otherwise | `src/baseline_models.py`, `baselines` section of the config | Yes |
| Training-size study: 25/50/100/200 (+400 for classification) and full set, seeds 42–44 | `sample_size_study` section; sizes not below the training size are replaced by the full set | Yes |
| Malignant is the positive class | `load_classification_data()` | Yes |
| Test labels never reach Kumo | query rows exclude the target; unit tests `test_kumo_never_sees_test_targets`, `test_kumo_rejects_target_column_in_query` | Yes |
| Datasets: 569 rows / 30 features / 212 malignant; 442 rows / 10 features / target 25–346; train/test 455/114 and 353/89 | scikit-learn loaders; checked by `check_manuscript_numbers.py` | Yes |
| Parameter counts 27.5M and 28.5M | counted from the loaded `small` checkpoints | Yes (same environment only) |

## 2. Claim-by-claim classification

Categories:

- **R-Kaggle**: reproduced in a different environment (author-run Kaggle notebooks) and identical in the original environment re-run.
- **R-local**: identical in the original-environment re-run only.
- **S**: verified against existing saved outputs (stored notebook outputs or `results/`) without being regenerated.
- **P**: partially verified.
- **N**: not verified.

| Claim (manuscript location) | Category | Notes |
|---|---|---|
| Kumo Tabular classification metrics: accuracy 0.989, precision 0.995, recall 0.976, F1 0.985, ROC-AUC 0.998 (abstract, Table 4) | R-Kaggle | ROC-AUC 0.9983 on Kaggle vs 0.9982 locally; same at reported precision |
| Logistic regression and random forest classification metrics (Table 4) | R-Kaggle | Identical |
| HistGradientBoosting classification metrics (Table 4) | P | Identical in the local re-run; different on Kaggle (accuracy 0.960 vs 0.967) |
| Pooled false negatives / false positives for Kumo 5/1, logistic regression 11/2, random forest 15/11 (Section V-A) | R-Kaggle | Identical |
| HistGradientBoosting false negatives / false positives 14/5 (Section V-A) | P | 18/5 on Kaggle |
| Kumo regression metrics: MAE 42.48, RMSE 52.88, R² 0.503 (abstract, Table 5) | R-Kaggle | Kaggle 42.479 / 52.880 / 0.5031 |
| Linear regression and random forest regression metrics (Table 5) | R-Kaggle | Identical |
| HistGradientBoosting regression metrics (Table 5) | P | Kaggle MAE 46.27 vs 45.85 |
| Interval coverage 93.3% ± 3.3%, width 174.2 ± 7.8; coverage range 87.6–95.5% (Section V-B) | R-Kaggle for mean coverage and width; R-local for the range | Kaggle 0.9326 ± 0.0328, width 174.24 |
| Point-estimate comparison, Table 6 | R-Kaggle | Kaggle mean-of-quantiles RMSE 52.3534 vs 52.3537 |
| Absolute-error correlations 0.93 (linear) and 0.76 (random forest) (Section V-B) | R-Kaggle | Identical |
| Absolute-error correlation 0.72 (HistGradientBoosting) (Section V-B) | P | 0.74 on Kaggle |
| Per-split accuracy and R² (Table 7) | R-local; S for Kaggle (seeds 42–44 seen) | Kaggle per-seed accuracy for Kumo and logistic regression matches for the seeds shown |
| Paired comparison: wins/ties/losses, mean advantages (Table 8, Section V-C) | R-Kaggle for W/T/L and mean advantage; R-local for p-values and intervals | p-values and intervals are recomputed from `results/` by `build_paper_assets.py` |
| No result significant after Holm adjustment; smallest adjusted p = 0.081; Wilcoxon minimum 0.0625 (Section V-C) | R-local | Derived from the identical per-split values |
| Training-size study, Kumo / logistic / linear / random forest (Tables 9–10, Fig. 8, Section V-D) | R-Kaggle | All match the means shown on Kaggle |
| Training-size study, HistGradientBoosting below the full size, incl. constant predictions at 25 rows (0.632, ROC-AUC 0.500) | R-Kaggle | Match |
| HistGradientBoosting at the full training size in the study | P | Kaggle 0.947 / 0.381 vs 0.959 / 0.406 |
| Error analysis: 38 cases, 4 misclassified by all models, Kumo's 6 errors (Section V-F) | S | From stored outputs of the local run. On Kaggle, Kumo's errors and probabilities match, but one case changes category because HistGradientBoosting differs |
| Runtimes 1.10 s / 0.63 s (Kumo) vs 0.019 s / 0.003 s (linear models), ratios ~60 and ~220 (Section V-E) | P | Local timings only. Re-run: 1.22 / 0.052 and 0.68 / 0.023 s. Kaggle: Kumo 0.41–0.52 s, ratios ~6–49. The manuscript states the range ~6–220 |
| Kumo time vs context size 0.46–1.11 s and 0.44–0.64 s (Section V-E) | P | Local hardware only |
| Kaggle results stated in Data and Code Availability (Section IX) | S | Transcribed from the public Kaggle outputs, recorded in `reproduction/kaggle_execution_evidence.md` |

No reported claim is in category N.

## 3. Original-environment re-run (1 November 2026)

Command: `python scripts/run_all.py --results-dir <scratch directory>`. The outputs are in `reproduction/`.

| Output compared with `results/` | Rows | Maximum absolute metric difference |
|---|---|---|
| Classification, five splits | 20 | 0 |
| Classification, training-size study | 72 | 0 |
| Regression, five splits | 20 | 0 |
| Regression, training-size study | 60 | 0 |

Mean fit plus prediction time per split, in seconds (saved run → re-run): Kumo classification 1.099 → 1.218, logistic regression 0.019 → 0.052, random forest 1.204 → 1.493, HistGradientBoosting 0.274 → 0.376. Kumo regression 0.627 → 0.678, linear regression 0.003 → 0.023, random forest 0.833 → 0.961, HistGradientBoosting 0.190 → 0.190.

## 4. Kaggle notebooks

Full transcription and manual re-check steps: `reproduction/kaggle_execution_evidence.md`.

| Notebook | Run record |
|---|---|
| Classification benchmark | Version 1 of 1, "65.7 second run - successful", GPU T4 x2 |
| Regression benchmark | Version 1 of 1, "92.9 second run - successful", GPU T4 x2 |
| Experimental analysis | Version 1 of 1, "126.7 second run - successful", GPU T4 x2; 20/20 classification and 20/20 regression model runs completed |

The HistGradientBoosting differences have not been traced to a cause. The scikit-learn version on Kaggle is not printed, so the version explanation given in the manuscript is stated as a presumption.

## 5. Manuscript corrections made on this evidence

- Runtime: the earlier "one to two orders of magnitude" was too strong for the Kaggle measurements (as low as ~6×). The abstract, Section V-E and the conclusion now state a ratio of about 6 to 220 across the hardware and runs measured.
- Data and Code Availability now reports the Kaggle execution, including the HistGradientBoosting difference.

## 6. Remaining limitations

- No third-party reproduction.
- Kaggle evidence was transcribed from the public pages; the notebooks were not re-run during this check.
- The scikit-learn version on Kaggle is unknown.
- Timing results depend on hardware and system load.
