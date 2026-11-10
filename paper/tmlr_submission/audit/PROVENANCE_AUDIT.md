# Provenance audit: original two-dataset experiments

Performed 10 November 2026 with read-only searches. Nothing was committed, pushed or changed in Git.

## What was searched

- **Working tree:** every file in the project, excluding `.venv/`, `.tools/` and the OpenML download cache. Each file was checked for a modification time earlier than the start of the Claude session (31 Oct 2026, 13:35 local time).
- **Git history:**
  - the current repository: 2 commits, the earliest `863db6c` on 9 Nov;
  - the pre-rewrite history in `.git-backup/`: earliest commit `2199c69`, 31 Oct 2026 16:27, all branches and remote-tracking refs.
- **Results:**
  - `results/classification/`, `results/regression/`, `results/figures/`;
  - `paper/final_submission/reproduction/`, including `kaggle_execution_evidence.md`;
  - the three notebooks and their stored outputs.
- **Session record:** the Claude Code session record (31 Oct – 10 Nov 2026), including:
  - the first directory listing of the project folder;
  - the original brief;
  - the commands that produced the results.

**Not searched:** anything outside this project folder, such as other folders on this machine, other accounts, Kaggle or Colab history, or any earlier tool or service you may have used.

## Findings

1. **No output of the author's original experiments exists in the repository or its history.**
   - At the start of the Claude session (31 Oct 2026, 13:35), `ls -la` of the project folder showed it **empty**. The folder was created at 13:18 that day.
   - No file in the working tree is older than the session.
   - Both Git histories begin after the session started.
   - The machine had no Python environment with PyTorch or Kumo Tabular at that time. Claude created `.venv` at 13:37.

   This does not show that the original experiments did not happen. It shows only that their outputs are **not available here**.

   **Author confirmation, 10 November 2026:** "I confirm that I do not have any surviving outputs, notebooks, CSV files, logs, screenshots, or backups from my original two-dataset experiments." So the original experiments and their initial analysis are recorded as the author's statement. They are **not independently verified**, and they are **not contradicted** by any evidence. Nothing was reconstructed in their place.
2. **Every case-study number in the manuscript comes from Claude's re-implementation.**
   - Source: `scripts/run_all.py`, written by Claude and run locally by Claude on 31 Oct 2026.
   - The outputs are in `results/` (files dated 14:18–14:20 local). They are byte-identical to the versions committed in `2199c69` (31 Oct, 16:27) and unchanged since.
3. **The author's Kaggle runs** (run date not recorded here; observed on 2 Nov 2026; the notebooks were added to the repository on 31 Oct) executed the notebooks Claude generated, under the author's account. They are an **author-executed reproduction of Claude's re-implementation**, not the author's original independent experiments. They match the local results except HistGradientBoosting.
4. **The paired significance analysis** (Nadeau–Bengio corrected t-test, Holm correction) was added by Claude on 1 Nov 2026 (`paper/build_paper_assets.py`), computed from the same `results/` files.
5. **The original brief (31 Oct)** documents the author's pre-existing methodology. It specifies datasets, models, metrics, train/test separation, context and query rows, no test labels for Kumo, and quantile handling. It describes train/test splitting ("stratified train/test splitting", "set random_state") and does not mention repeated splits, a training-size study, interval evaluation or significance tests.

## Provenance table

**Basis** uses three classes:
- **Author statement:** rests only on the author's word.
- **Author statement + repository evidence:** the author's word, partly supported by repository evidence.
- **Repository-verified:** shown directly by files, Git history or the session record.

### The author's original contributions (completed before Claude's involvement)

| Contribution | Basis | Evidence found | Source |
|---|---|---|---|
| Research idea and hypothesis | Author statement + repository evidence | The idea and hypothesis are already set out in the author's brief of 31 Oct, the first message of the Claude session. Their earlier origin rests on the author's statement | Session record; author statement (10 Nov) |
| Initial benchmark methodology | Author statement + repository evidence | The brief specifies datasets, train/test splitting, context and query separation, no test labels for Kumo, and quantile handling. That it was developed earlier, independently, is the author's statement | Session record; author statement |
| Original experiments on the two datasets | **Author statement** | None. The author confirms no outputs, notebooks, CSV files, logs, screenshots or backups survive. Not independently verified; nothing contradicts it | Author statements (10 Nov) |
| Evaluation metrics and baseline selection | Author statement + repository evidence | The brief lists the metrics (accuracy, precision, recall, F1, ROC-AUC, confusion matrix; MAE, RMSE, R²) and the three baselines | Session record; author statement |
| Initial analysis of the original results | **Author statement** | None surviving (author's confirmation). Not independently verified; nothing contradicts it | Author statements (10 Nov) |

### Claude's later work and the author-executed reproduction

| Contribution | Basis | Evidence found | Source |
|---|---|---|---|
| Re-implementation: code, notebooks, tests (Claude) | Repository-verified | Files created in the session; the folder was empty before | Session record; Git `2199c69` |
| All reported two-dataset results: five splits, all case-study metrics (Claude) | Repository-verified | `run_all.py` executed by Claude on 31 Oct; outputs byte-identical to the first commit | Session record; `results/`; Git `2199c69` |
| Training-size study and quantile-interval evaluation (Claude) | Repository-verified | First appear in Claude's code on 31 Oct; absent from the brief | Session record; `sample_size_study.csv` |
| Paired corrected tests and Holm adjustment for the case studies (Claude) | Repository-verified | Added 1 Nov, in response to the author's instruction not to claim significance without evidence | Session record; `paper/build_paper_assets.py` |
| Kaggle runs of Claude's notebooks (executed by the author) | Repository-verified | Three public notebooks, "Version 1 of 1", Tesla T4; values transcribed 2 Nov. These reproduce the re-implementation, not the original experiments | `paper/final_submission/reproduction/kaggle_execution_evidence.md` |
| OpenML extension: design, code, runs, analysis, interpretation (Claude). The author chose the OpenML extension and the extra libraries | Repository-verified | Session record; `ANALYSIS_PLAN.md`; `results/openml_benchmark/` | Session record; results |
| Manuscript text, tables and figures (Claude, following the author's instructions) | Repository-verified | Session record; IEEE disclosure approved 2 Nov | Session record |

## Origin of each case-study number in the TMLR manuscript

All of these come from Claude's local run on 31 Oct 2026. Where the Kaggle run is noted, the author's Kaggle notebooks give the same value.

| Manuscript value | Location | Source file | Kaggle (author-run) |
|---|---|---|---|
| Kumo accuracy 0.989 vs logistic regression 0.977 (breast cancer) | Section 5.5; Appendix B table | `results/classification/summary.csv` | Same (0.9895 / 0.9772) |
| Smallest adjusted p = 0.081 | Section 5.5; Appendix B paired table | `results/classification/results_by_seed.csv` via `paper/build_paper_assets.py` (Claude, 1 Nov) | Not computed on Kaggle |
| R² 0.503 vs 0.496 (diabetes) | Section 5.5; Appendix B table | `results/regression/summary.csv` | Same (0.5031 / 0.4959) |
| R² 0.429 vs 0.075 with 25 training rows | Section 5.5; Appendix B training-size table | `results/regression/sample_size_study.csv` | Same (0.429 / 0.075) |
| 90% interval coverage 93.3% | Section 5.5; Appendix B figure | `results/regression/summary.csv` (`interval_coverage_mean` 0.9326) | Same (0.9326) |
| HistGradientBoosting accuracy 0.967 (local) vs 0.960 (Kaggle) | Appendix B | `summary.csv`; Kaggle evidence | Differs, as disclosed |
| Appendix B tables and figures (`original_*`) | Appendix B | Copied from `paper/final_submission/latex_source/` by `build_tmlr_assets.py`; generated from `results/` | — |

## Consequences for the submission

- The manuscript must not say or imply that the author's original results are reproduced. Section 4 now says that all case-study numbers come from the re-implementation, and that the outputs of the earlier experiments are not available.
- In the appendix, "re-run" and "cloud execution" now say they refer to "the same code" and "the same notebooks".
- The footnote keeps the author's original contributions as the author's statement, and attributes the reported two-dataset results to Claude's re-implementation.
- The author confirmed that no original outputs survive. The original experiments and their initial analysis therefore stay documented as the author's statement: not classified as verified, not described as absent, and not reconstructed. No number from Claude's re-implementation is attributed to them.
