# Git audit

Taken on 10 November 2026 with read-only commands (`git for-each-ref`, `git reflog`, `git ls-remote`, `git diff`, `git status`). Nothing was pushed, fetched into a branch, rebased, reset or deleted during the TMLR work.

## Remote

`origin` = `https://github.com/noyonahmedjoy/Kumo-Tabular.git`

| Remote ref (`git ls-remote origin`) | Commit |
|---|---|
| `HEAD` | `863db6cbaa87de1507b45ef428f463b67ae0133b` |
| `refs/heads/main` | `863db6cbaa87de1507b45ef428f463b67ae0133b` |

The remote has only `main`. The branch `paper/ieee-access-submission`, which was pushed on 2 November 2026, no longer exists on the remote.

## Local refs

| Ref | Commit | Notes |
|---|---|---|
| `refs/heads/main` | `863db6c` | Same as `origin/main` |
| `refs/heads/paper/tmlr-submission` (checked out) | `1a34db972fb4b6c03e33e8eb6bf3364d0de17093` | One commit ahead of `main`; **not pushed** |
| `refs/remotes/origin/main` | `863db6c` | Reflog: "update by push", 2026-11-09 19:21:07 +0600 |

Local reflog:
- `863db6c`: "commit (initial)", 2026-11-09 19:18:04. The repository was re-initialized; this was not done in this session.
- 2026-11-09 22:10:04: checkout to `paper/tmlr-submission`.
- `1a34db9`: commit "Pre-register OpenML benchmark: suite, code and analysis plan", 2026-11-09 22:10:05.

## History before the re-initialization (`.git-backup/`, ignored by `.gitignore`)

The `.git-backup/` folder holds the previous repository. In it:

- `main` = `2dbca1a`, the merge pushed on 2 November with approval.
- `paper/ieee-access-submission` = `3809288`.
- The remote-tracking refs point to **different** commits with the same messages: `origin/main` = `304dbb6` and `origin/paper/ieee-access-submission` = `d181bf7`. The remote history was therefore rewritten after the approved push of 2 November.
- The remote history was then replaced by the single commit `863db6c` (19:21 on 9 November).

None of these rewrites were performed by the assistant.

## What the public `main` (`863db6c`) contains

- 135 tracked files.
- This includes the IEEE Access LaTeX template files in `paper/final_submission/latex_source/`, among them 16 Type 1 font files (Times LT Std, Formata, Giovanni Std).
- `audit/FONT_LICENSE_AUDIT.md` documents their licence notices, which prohibit or restrict redistribution. They are therefore publicly redistributed at present. They are not needed for the TMLR build, which uses Latin Modern.
- They have not been modified or deleted, pending the author's decision.

## Committed but not pushed: `main..paper/tmlr-submission` (`1a34db9`)

12 files, +2,699 lines, no deletions:

```
.gitignore                                      |    2 +
configs/openml_suite.json                       | 1826 +
paper/tmlr_submission/audit/ANALYSIS_PLAN.md    |   46 +
paper/tmlr_submission/audit/REPOSITORY_AUDIT.md |   86 +
scripts/analyze_openml_benchmark.py             |  195 +
scripts/run_openml_benchmark.py                 |   79 +
scripts/select_openml_suite.py                  |  122 +
src/baseline_models.py                          |   32 +
src/evaluation.py                               |   27 +
src/openml_benchmark.py                         |   95 +
src/openml_data.py                              |   78 +
tests/test_openml_benchmark.py                  |  111 +
```

The commit time, 22:10:05, is after the `wdbc` verification run finished (22:08:23) and before the second dataset finished (22:10:32). The analysis plan states that the `wdbc` result was the only result seen before the plan was committed.

## Not committed (working tree)

**Modified tracked files** (5 files, +39 / −2, as of the pre-commit audit on 10 November 2026):

| File | Change |
|---|---|
| `.gitignore` | Two exclusions added: console captures and `.bbl` build output |
| `configs/openml_suite.json` | `excluded_at_load` entry for `forest_fires` |
| `paper/tmlr_submission/audit/ANALYSIS_PLAN.md` | "Correction" note and "Deviations" section added; the original plan text is unchanged |
| `scripts/run_openml_benchmark.py` | Logs a dataset that fails to load as "NOT RUN" and continues |
| `src/openml_data.py` | `load_suite()` skips `excluded_at_load` entries |

**Untracked:**

82 untracked files in total.

- `results/openml_benchmark/`: raw results (21 CSV files), combined results, run log, environment, validation report and analysis.
  - `console_output.txt` and `console_output_resume.txt` contain local file paths. They are now git-ignored, kept locally and excluded from the ZIP.
- `scripts/validate_openml_results.py`
- `paper/tmlr_submission/`:
  - `manuscript/`: LaTeX source, style files, the unused template copy `template_main_reference.tex`, 12 tables, 7 figures and the built `main.pdf`. The LaTeX auxiliary files are git-ignored.
  - `supplementary/anonymized_code_and_results.zip`
  - `AI_ASSISTANCE_DISCLOSURE.md`, `REPRODUCIBILITY.md`, `FINAL_REPORT.md`
  - `build_tmlr_assets.py`, `check_anonymity.py`, `check_tmlr_numbers.py`, `make_supplementary.py`, `verify_statistics.py`
  - `audit/`: `STATISTICAL_AUDIT.md`, `EXPERIMENTAL_GAP_ANALYSIS.md`, `FONT_LICENSE_AUDIT.md`, `REFERENCE_AUDIT.md`, `PROVENANCE_AUDIT.md` and this file

No tracked file is deleted.

**IEEE materials:** `paper/final_submission/` is unchanged (`git status` shows no change under it).

## Pending remote operations (none performed; each needs the author's approval)

1. Whether to push `paper/tmlr-submission`. Pushing it to the public repository under the author's name would let reviewers link the anonymous submission to the author. Under double-blind review it is safer to keep it private until a decision is made.
2. Whether and how to remove the licence-restricted font files from the public `main`. A normal deletion commit stops further distribution from the current tree, but the files stay in history. Removing them from history requires a rewrite, which the author has not authorized.
3. Whether to restore the IEEE branch on the remote.

## Local changes made on 10 November 2026 (no commit, no remote operation)

- `.gitignore`: two lines appended, `results/openml_benchmark/console_output*.txt` and `paper/tmlr_submission/manuscript/*.bbl`, so that console captures with local paths and LaTeX build output cannot be staged by accident.
- Three stray build-output files that I had created (`paper/tmlr_submission/manuscript/r.txt`, `r1.txt`, `rb.txt`, which are pdfLaTeX and BibTeX console captures containing local paths) were moved out of the project into the session scratchpad. They were never tracked.
- Note: `.gitignore` on `main` already lists the IEEE template files (`paper/final_submission/latex_source/t1*`, `ieeeaccess.cls`, `spotcolor.sty`, logos), but all 45 of them are tracked in `863db6c`. `.gitignore` does not affect files that are already tracked.
