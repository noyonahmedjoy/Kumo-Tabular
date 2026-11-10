# Reproducibility and execution report: OpenML benchmark (TMLR submission)

Status on 9 November 2026: all experiments finished. The manuscript has not been submitted, and it has not been reviewed or accepted.

## 1. Execution record, dataset by dataset

Source: `results/openml_benchmark/run_log.txt`, the raw files `results/openml_benchmark/raw/*.csv`, and `configs/openml_suite.json`. Each completed dataset has 80 result rows (10 official OpenML folds × 8 models). "Wall time" is the runner's time for all ten folds and all eight models, including data loading.

| OpenML data | Task | Dataset | Type | Rows | Features | Classes | Status | Result rows | Rows `ok` | Wall time |
|---|---|---|---|---|---|---|---|---|---|---|
| 11 | 11 | balance-scale | classification | 625 | 4 | 3 | completed | 80 | 80 | 80 s |
| 18 | 18 | mfeat-morphological | classification | 2000 | 6 | 10 | completed | 80 | 80 | 280 s |
| 22 | 22 | mfeat-zernike | classification | 2000 | 47 | 10 | completed | 80 | 80 | 2830 s |
| 37 | 37 | diabetes | classification | 768 | 8 | 2 | completed | 80 | 80 | 94 s |
| 54 | 53 | vehicle | classification | 846 | 18 | 4 | completed | 80 | 80 | 115 s |
| 1049 | 3902 | pc4 | classification | 1458 | 37 | 2 | completed | 80 | 80 | 158 s |
| 1050 | 3903 | pc3 | classification | 1563 | 37 | 2 | completed | 80 | 80 | 147 s |
| 1063 | 3913 | kc2 | classification | 522 | 21 | 2 | completed | 80 | 80 | 96 s |
| 1068 | 3918 | pc1 | classification | 1109 | 21 | 2 | completed | 80 | 80 | 108 s |
| 1462 | 10093 | banknote-authentication | classification | 1372 | 4 | 2 | completed | 80 | 80 | 156 s |
| 1464 | 10101 | blood-transfusion-service-center | classification | 748 | 4 | 2 | completed | 80 | 80 | 85 s |
| 1494 | 9957 | qsar-biodeg | classification | 1055 | 41 | 2 | completed | 80 | 80 | 133 s |
| 1510 | 9946 | wdbc | classification | 569 | 30 | 2 | completed | 80 | 80 | 127 s |
| 40982 | 146817 | steel-plates-fault | classification | 1941 | 27 | 7 | completed | 80 | 80 | 364 s |
| 40994 | 146819 | climate-model-simulation-crashes | classification | 540 | 20 | 2 | completed | 80 | 80 | 116 s |
| 44957 | 361235 | airfoil_self_noise | regression | 1503 | 5 | — | completed | 80 | 80 | 78 s |
| 44959 | 361237 | concrete_compressive_strength | regression | 1030 | 8 | — | completed | 80 | 80 | 74 s |
| 44960 | 361617 | energy_efficiency | regression | 768 | 8 | — | completed | 80 | 80 | 71 s |
| 44962 | 361618 | forest_fires | regression | 517 | 12 | — | **not run (excluded at load)** | 0 | 0 | — |
| 44970 | 361621 | QSAR_fish_toxicity | regression | 908 | 6 | — | completed | 80 | 80 | 123 s |
| 44972 | 361250 | red_wine | regression | 1599 | 11 | — | completed | 80 | 80 | 74 s |
| 44994 | 361622 | cars | regression | 804 | 17 | — | completed | 80 | 80 | 70 s |

Summary:
- **Selected:** 22 datasets (15 classification, 7 regression).
- **Completed:** 21 datasets (15 classification, 6 regression).
- **Failed model runs:** 0.
- **Skipped datasets:** 1, `forest_fires`.
- **Rows:** 1,680 result rows, all with status `ok`.
- **Wall time:** the per-dataset times sum to 5,379 s (about 1.5 h), on the hardware in Section 4.
- **`mfeat-zernike`:** this one dataset took 53% of the total time. Over its ten folds, the time per model was HistGradientBoosting 1,113 s, CatBoost 841 s and Kumo Tabular 650 s.

### Run history

1. **22:05–22:08: smoke test and verification run on `wdbc`.**
   - This run was used to check the pipeline end to end.
   - Its result file was kept as the final `wdbc` result, as recorded in `audit/ANALYSIS_PLAN.md` under Deviations.
2. **22:08–23:32: main run.**
   - The run completed 17 more datasets.
   - It then stopped with an exception while loading `forest_fires`. The OpenML metadata for that dataset reports 0 symbolic features, but its `month` and `day` columns contain text, and `pd.to_numeric(errors="raise")` rejected them.
   - No result for `forest_fires` was ever produced.
   - The traceback is in the local file `results/openml_benchmark/console_output.txt`. That file is excluded from the anonymized archive because it contains local file paths.
3. **Exclusion.** `forest_fires` was excluded under the pre-specified inclusion criterion "numerical features only".
   - The exclusion was recorded in `configs/openml_suite.json` (`excluded_at_load`, with its reason and date).
   - It was also recorded in `audit/ANALYSIS_PLAN.md` (Deviations).
   - The runner was changed so that a dataset that cannot be loaded is logged as "NOT RUN" and the run continues.
4. **23:35–23:39: resumed run.**
   - The 18 completed datasets were skipped, because their result files were already complete.
   - The remaining three regression datasets were run.
   - All 21 result files were then combined into `results_by_fold.csv`.

### Consequence of the exclusion

The regression analysis has n = 6 datasets instead of 7. The smallest attainable exact two-sided Wilcoxon p-value is 0.031 for n = 6, and 0.0156 for n = 7. With a Holm correction over seven comparisons, no regression comparison can reach 0.05 in either case. The regression results are therefore descriptive, as the analysis plan had already anticipated for n = 7.

## 2. Validation of the result files

`python scripts/validate_openml_results.py` writes its output to `results/openml_benchmark/validation_report.json`. It reported 21 of 21 files valid. The checks were:

- For every dataset: the expected eight models, folds 0–9 exactly once per model, and no failed rows.
- In each fold, training plus test size equals the dataset size, and the test sizes sum to the dataset size.
- The OpenML data and task IDs match the suite.
- Every metric is finite and inside its valid range: accuracy, F1 and AUC in [0, 1]; log loss ≥ 0; RMSE and MAE ≥ 0; R² ≤ 1.
- No runtime is negative.
- `results_by_fold.csv` is identical to the concatenation of the raw files.

Row-level split checks run inside the loader during the benchmark (`src/openml_data.py`): a fold whose training and test indices overlap is rejected, and the test folds must cover every row exactly once.

## 3. Independent verification of the statistics

`python paper/tmlr_submission/verify_statistics.py` recomputes every statistic in `results/openml_benchmark/analysis/analysis.json` from the raw per-fold rows. It uses its own implementation:

- Means use `math.fsum`.
- Ranks are averaged over ties.
- The Friedman statistic is hand-coded, with a tie correction.
- The exact two-sided Wilcoxon p-values are computed by enumerating all sign assignments.
- The Holm step-down procedure is hand-coded, with monotonicity enforced.

Result: **217 of 217 statistics match.**

The first run of the verifier reported 7 mismatching accuracy ranks. They came from naive floating-point summation in the verifier: its fold means differed in the last bit, which broke ties that are exact in the data. Using `math.fsum` in the verifier fixed this. The analysis script was not changed.

`python paper/tmlr_submission/check_tmlr_numbers.py` compares 53 numbers quoted in the manuscript with the result files. Result: **53 of 53 match.**

## 4. Tests and environment

- `python -m pytest`: 45 of 45 tests pass (re-run on 9 November 2026 after the benchmark). On this machine pytest needs `--basetemp` set to a writable folder; without it, two tests that use `tmp_path` fail with a permission error from the sandbox, which is not a code failure. Eight of the tests cover the OpenML pipeline: split parsing, rejection of overlapping folds, metric values checked by hand, probability mapping onto the full class set, a train/test leakage check with a stub model, and the path where Kumo is unavailable.
- The exact software versions are in `results/openml_benchmark/environment.json`.
- Hardware: NVIDIA GeForce GTX 980M (compute capability 5.2, so float32 inference) and an Intel Core i7-6700HQ, running Windows 10.
- Kumo Tabular: `small` checkpoint, default settings (8 estimators). The regression point prediction is the `q500` quantile.

## 5. Reproduce

```bash
pip install -r requirements.txt xgboost lightgbm catboost
python -m pytest
python scripts/select_openml_suite.py
python scripts/run_openml_benchmark.py
python scripts/validate_openml_results.py
python scripts/analyze_openml_benchmark.py
python paper/tmlr_submission/verify_statistics.py
python paper/tmlr_submission/build_tmlr_assets.py
python paper/tmlr_submission/check_tmlr_numbers.py
```

`select_openml_suite.py` queries the live OpenML API. If OpenML metadata changes, the selection could differ, so `configs/openml_suite.json` is the authoritative record of the suite. The runner resumes: a dataset whose result file is complete is skipped.

Kumo Tabular is expected to be bit-identical across re-runs on the same hardware, because its generator is seeded. This was checked in the original two-dataset study, not by re-running the OpenML benchmark. Timings and float16 results on other GPUs will differ.

## 6. Limits of this record

- **Timing of the analysis plan.**
  - The plan was written at 22:09:50 and committed to a **local** Git repository at 22:10:05 (commit `1a34db9`, 9 November 2026).
  - Before that, the complete `wdbc` results had been seen: the 10-fold means of all eight models, printed at 22:08:35. Results for the other 20 datasets did not exist yet.
  - The session record does not show that the primary metrics were chosen before the `wdbc` output was seen.
  - The commit has not been pushed, so no third party holds a timestamp for it. This is therefore not a formal preregistration, and the manuscript says so in Sections 4 and 8.
- The benchmark was run once. Run-to-run variation of the CPU baselines was not measured. HistGradientBoosting and the other multithreaded baselines may differ slightly between machines and library versions.
- Baselines use library defaults with no tuning. Kumo Tabular uses one checkpoint size.
