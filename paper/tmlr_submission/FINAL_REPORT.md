# TMLR preparation: final report

Updated 10 November 2026 (second revision: author contribution history).

**Status:** prepared draft, **not ready for submission** (see section 6). Nothing has been submitted to TMLR, and nothing has been reviewed, accepted or published. No remote Git operation was performed. All TMLR work is uncommitted on branch `paper/tmlr-submission`, except the local commit `1a34db9`.

## 1. Final submission files

| File | Purpose | State |
|---|---|---|
| `paper/tmlr_submission/manuscript/main.pdf` | Anonymized manuscript for the OpenReview upload | 16 pages: main text pp. 1–11, references pp. 11–13, appendix pp. 13–16. Official TMLR style. 0 LaTeX errors, 0 overfull boxes, 0 undefined references, 44/44 references cited. No PDF author, creator, producer or date metadata |
| `paper/tmlr_submission/supplementary/anonymized_code_and_results.zip` | Anonymized supplementary material | 113 files, 2.57 MB, archive integrity check passed. Contains code, tests, configuration, notebooks, raw and combined results, run log, environment record, analysis outputs, the analysis plan (with its correction note) and the verification script. PDF metadata stripped; no console logs |

Everything else under `paper/tmlr_submission/` is internal documentation and is **not** uploaded.

## 2a. Contribution history and AI disclosure (second revision, 10 November 2026)

**Basis.** The author's confirmed contribution history, given on 10 November 2026, was checked against three records:
- the Claude Code session record, 31 October – 10 November 2026;
- the written brief of 31 October;
- the IEEE disclosure of 2 November.

**First-page footnote (`manuscript/main.tex`).** It is now titled "Use of large language models and division of work" and has two labelled parts.
- *Original study (author)*: the five contributions the author completed independently before Claude's involvement, listed in full as confirmed:
  - the research idea and hypothesis;
  - the initial methodology;
  - the metrics and baselines;
  - the experiments on the two datasets;
  - the initial analysis.
- *AI-assisted work*:
  - Claude re-implemented the two-dataset study and produced the two-dataset results printed in the paper. It added the repeated splits, the training-size study, the interval evaluation and the corrected tests.
  - For the OpenML extension, the author chose to add OpenML datasets and chose the additional libraries, from Claude's options. Claude did the design, code, runs, analysis, interpretation and reference checks.
  - Claude drafted and revised the text, tables and figures.
  - ChatGPT's role is stated separately.
- No dates were added. None are recorded for the original work.

**What was deliberately kept, and why:**
- **The case-study numbers.** The paper's two-dataset numbers come from Claude's re-implementation and runs, not from the author's earlier runs. Those earlier outputs are not in this repository. The footnote says this, so that Claude's work is not attributed to the author. Reporting the author's own results instead would mean adding them to the repository and the paper, which is a new decision.
- **The manuscript text.** It is attributed to Claude. The session record and the IEEE disclosure the author approved on 2 November both say Claude drafted the text. The author's new statement, that the initial manuscript and documentation "involved both", is recorded in `AI_ASSISTANCE_DISCLOSURE.md`. If the author wrote specific passages, the author should name them so they can be credited.
- **Responsibility.** No statement of review or responsibility was added, because the author has not yet reviewed this version.

**`AI_ASSISTANCE_DISCLOSURE.md`** was rewritten as a contribution record:
- (A) the author's original study;
- (B) AI-assisted work, split into the two-dataset re-implementation and the OpenML extension;
- (C) manuscripts and documentation;
- (D) ChatGPT.

It also contains the policy check and the remaining author actions.

**TMLR policy check (editorial policies and FAQ, re-read 10 November 2026).**
- The first-page footnote is present, the tools are named, and no AI is listed as an author.
- Authors are responsible for all content (see action 2 in section 6).
- Under TMLR's "human-sourced" expectation:
  - the original idea, hypothesis and methodology are human-sourced;
  - the OpenML extension is not. Its design, results and interpretation are the paper's main claims.
- TMLR also says submissions that appear largely LLM-generated will be "closely examined".

This remains a submission blocker that only the author can decide.

## 2b. Provenance audit of the original two-dataset experiments (10 November 2026)

Full record: `audit/PROVENANCE_AUDIT.md`.

**Result.** No output of the author's original experiments exists in this repository or its Git history (current or `.git-backup/`).
- The project folder was empty when the Claude session began (31 Oct 2026, 13:35).
- No file predates the session.

The original experiments and the initial analysis are therefore documented only by the author's statement. Areas outside this folder, such as other folders, Kaggle or Colab history, or other tools, were not searched.

**Where the case-study numbers come from.**
- All of them (accuracy 0.989 vs 0.977, adjusted p 0.081, R² 0.503 vs 0.496, R² 0.429 vs 0.075 at 25 rows, coverage 93.3%, and the Appendix B tables and figures) come from Claude's local run of its re-implementation on 31 Oct 2026. Those output files are byte-identical to the first commit.
- The author's Kaggle runs reproduced them (except HistGradientBoosting), but those runs executed Claude's notebooks. They are not the original experiments.

**Changes:**
- **Manuscript, Section 4:** the case studies are now described as a re-implementation, with additional analyses, of the author's earlier study. The text says that all case-study numbers come from the re-implementation, that the earlier outputs are not available, and that no claim of reproducing them is made.
- **Manuscript, Appendix B:** the re-run and the cloud execution are now explicitly "of the same code" and "of the same notebooks".
- **`AI_ASSISTANCE_DISCLOSURE.md`:**
  - the original experiments and initial analysis are marked "no artifact found";
  - the Kaggle runs moved from "original study" to "AI-assisted work", as an author-executed reproduction of the re-implementation;
  - a provenance section and an optional action (supply original outputs) were added.
- **Footnote:** unchanged. It already attributes the reported two-dataset results to Claude's re-implementation, and the author's original contributions as the author confirmed them.

**Author confirmation (10 November 2026).** No outputs, notebooks, CSV files, logs, screenshots or backups of the original experiments survive. The provenance audit and the disclosure now give every contribution an evidence basis:
- **Author statement:** the original experiments and the initial analysis. Not independently verified; nothing contradicts them.
- **Author statement + repository evidence:** the idea, the methodology, and the metrics and baselines. All three appear in the 31 October brief.
- **Repository-verified:** all of Claude's work and the Kaggle runs.

The five original contributions are kept in full. Nothing was reconstructed, and no number from Claude's re-implementation is attributed to the original experiments. The manuscript wording needed no change. It already says that the earlier outputs are not available and that no claim of reproducing them is made, which neither verifies nor denies the original experiments.

**Anonymity note.** The manuscript refers to "the author" in the singular, in the footnote and in Section 4. This reveals a single-author paper, but not the author's identity.

## 2. Changes made earlier on 10 November 2026

1. **Wording on when the analysis plan was written.**
   - The session record shows the sequence: the complete `wdbc` results (all eight models, all metrics) were seen at 22:08:35; the plan was written at 22:09:50 and committed locally at 22:10:05; results for the other datasets came later. There is no record that the primary metrics were chosen before the `wdbc` output was seen.
   - The manuscript no longer says the analysis was specified "before the results were inspected", and no longer uses "pre-specified":
     - abstract, introduction, results and conclusion: changed to "planned" wording;
     - Section 4: states the actual sequence and that the plan is not a formal preregistration;
     - Section 8: new paragraph "Timing of the analysis plan";
     - sensitivity table caption: changed to "Planned".
   - `ANALYSIS_PLAN.md` keeps its original text and has a dated correction note. `REPRODUCIBILITY.md` was corrected.
2. **`forest_fires`.** The exclusion stays as the plan's selection criterion requires. No new experiment was run. Section 8 now states it as a limitation: the metadata was wrong, the dataset was excluded after 17 other datasets had results, the regression suite dropped from seven to six datasets, and the dataset was not evaluated with its text columns encoded or removed.
3. **OpenML-CTR23 reference.**
   - Verified against the official OpenML study page (study 353), which carries the authors' own citation.
   - The entry was **missing an author** (L. Harutyunyan). The author is now added, and the forum URL is now in the entry.
   - The 9 November pass had wrongly marked this entry as verified.
   - The Bischl et al. author list was re-checked against arXiv and is correct.
4. **AI disclosure.** The first-page footnote was rewritten to match the session record:
   - It now attributes the extended design, the statistical analysis plan, the interpretation and conclusions, and all drafted text and figures to Claude.
   - It states that your role in the extension was choosing between options Claude proposed.
   - It **no longer says you reviewed the results and take full responsibility**, because you have not yet reviewed this version.
   - "Independent implementation" of the statistics now reads "second, separately written implementation", with a note that the same AI wrote both.
   - "Independent evaluation" is now defined as independent of the model's developers.
5. **Anonymity.** The checks now also cover:
   - any e-mail address;
   - PDF date metadata, whose time zone hinted at your location, in the PDF and in every PDF inside the ZIP.

   A time-zone mention and a commit hash were removed from the analysis plan copy in the ZIP.
6. **Stray files.** Three console captures with local paths (`manuscript/r.txt`, `r1.txt`, `rb.txt`, created by me) were moved out of the project into the session scratchpad. `.gitignore` now excludes `results/openml_benchmark/console_output*.txt` and `manuscript/*.bbl`.

## 3. Final checks (all run after the last edit)

| Check | Result |
|---|---|
| PDF build: pdflatex, bibtex, then pdflatex ×3 from clean auxiliary files | 0 errors, 0 overfull, 0 undefined, 2 underfull (cosmetic). BibTeX: 1 warning (`arik2021tabnet` issue number not printed; the data are correct) |
| `check_tmlr_numbers.py` | 53/53 manuscript numbers match the result files |
| `check_anonymity.py` on the PDF (text, links, metadata, XMP, dates) | 0 hits |
| `check_anonymity.py` on the ZIP (contents, file names, comments, embedded PDF metadata) | 0 hits |
| Wider manual scan of the PDF and ZIP (any e-mail address, GitHub or Kaggle link, drive or home paths, "IEEE", self-reference phrases, time zones, commit hashes) | No identifying hit. Remaining matches: the citation of an IEEE journal paper by other authors; NVIDIA's own GitHub and Hugging Face links; generic "runs on Kaggle, Colab or locally" instructions in the notebooks |
| ZIP integrity (`zipfile.testzip`) | OK |
| `validate_openml_results.py` | 21/21 files valid; 1,680 rows; combined file identical to the raw files |
| `verify_statistics.py` | 217/217 statistics reproduced |
| `pytest` (with `--basetemp` set to a writable folder) | 45/45 pass |

## 4. Generated files with local paths or console output

| File | Content | Handling |
|---|---|---|
| `results/openml_benchmark/console_output.txt` | Console capture of the first run, including the `forest_fires` traceback, with local paths | Keep locally as evidence; git-ignored; not in the ZIP |
| `results/openml_benchmark/console_output_resume.txt` | Console capture of the resumed run | Same |
| `paper/tmlr_submission/manuscript/main.log`, `.aux`, `.out`, `.blg`, `.bbl` | LaTeX build output; `main.log` contains local paths | Git-ignored; not in the ZIP |
| `results/openml_benchmark/analysis/*.pdf`, `manuscript/figures/*.pdf` | Matplotlib PDFs whose creation date includes the time zone | Fine inside your own repository. Stripped in the ZIP. The figures do not carry it into `main.pdf` |
| `results/openml_benchmark/run_log.txt` | Timestamps (local time, no time zone) and dataset progress | No paths. Included in the ZIP |

## 5. Proposed Git staging list (nothing staged or committed)

**Modified tracked files:**
```
.gitignore
configs/openml_suite.json
paper/tmlr_submission/audit/ANALYSIS_PLAN.md
scripts/run_openml_benchmark.py
src/openml_data.py
```
**New files:**
```
scripts/validate_openml_results.py
results/openml_benchmark/raw/                (21 CSV files)
results/openml_benchmark/results_by_fold.csv
results/openml_benchmark/run_log.txt
results/openml_benchmark/environment.json
results/openml_benchmark/validation_report.json
results/openml_benchmark/analysis/           (analysis.json, dataset_means.csv, 3 figure PDFs)
paper/tmlr_submission/AI_ASSISTANCE_DISCLOSURE.md
paper/tmlr_submission/FINAL_REPORT.md
paper/tmlr_submission/REPRODUCIBILITY.md
paper/tmlr_submission/audit/                 (EXPERIMENTAL_GAP_ANALYSIS, FONT_LICENSE_AUDIT, GIT_AUDIT, PROVENANCE_AUDIT, REFERENCE_AUDIT, STATISTICAL_AUDIT)
paper/tmlr_submission/build_tmlr_assets.py
paper/tmlr_submission/check_anonymity.py
paper/tmlr_submission/check_tmlr_numbers.py
paper/tmlr_submission/make_supplementary.py
paper/tmlr_submission/verify_statistics.py
paper/tmlr_submission/manuscript/main.tex
paper/tmlr_submission/manuscript/references.bib
paper/tmlr_submission/manuscript/sections/    (8 .tex files)
paper/tmlr_submission/manuscript/tables/      (12 generated .tex files)
paper/tmlr_submission/manuscript/figures/     (7 generated figures)
paper/tmlr_submission/manuscript/tmlr.sty, tmlr.bst, fancyhdr.sty, math_commands.tex, TMLR_STYLE_LICENSE
```
The TMLR style files are Apache-2.0 (`TMLR_STYLE_LICENSE`), and no font files are included.

**Excluded:**
- **Console logs:** `results/openml_benchmark/console_output*.txt`.
- **LaTeX build output:** `main.aux`, `main.log`, `main.out`, `main.blg`, `main.bbl`.
- **Caches and local folders:** `__pycache__/`, `.pytest_cache/`, `data/` caches, `.tools/` (TinyTeX), `.git-backup/`, `.venv/`.
- **Credentials:** none exist in the working tree.
- **IEEE materials:** all of `paper/final_submission/`, including the commercial font files. Nothing there is modified.
- **Recommended exclusions:** `manuscript/main.pdf` and `supplementary/anonymized_code_and_results.zip`. They can be regenerated from the staged sources. Committing the anonymized submission files into a repository under your name would link them to you if the branch were ever pushed during review.
- **Also excluded (pre-commit audit, 10 Nov):** `manuscript/template_main_reference.tex`. It is an unused copy of the official template example, and the build does not reference it. It contains only the template's third-party example author e-mails.

## 5a. Additional TMLR policy finding: IEEE Access

TMLR's Editorial Policies ("Dual Submissions and Originality", read 10 November 2026) say:
- TMLR accepts only original contributions that don't reuse the authors' own prior work.
- There must be no reuse of text, figures or results with any paper published, accepted or submitted in parallel at another archival, peer-reviewed venue.
- Overlap is allowed only with non-archival venues or preprint servers.

IEEE Access is archival and peer-reviewed. The TMLR manuscript reuses the IEEE study's results and figures as its case studies (Section 5.5, Appendix). So:
- If the IEEE Access manuscript has been, or will be, submitted, accepted or published, **the TMLR submission would violate this policy as written.**
- `paper/final_submission/Submission_Readiness_Report.md` (2 November) records it as not submitted. Its current status is known only to you.
- The public GitHub copy of the IEEE PDF is not a declared preprint server either.

## 6. Unresolved decisions and exact actions for you

**Blocking submission:**

1. **IEEE Access versus TMLR: choose one.**
   - If you pursue TMLR, do not submit the IEEE Access version. If it is already submitted, withdraw it before submitting to TMLR, or do not submit to TMLR.
   - Ask the TMLR editors if unsure.
   - Optionally, post the IEEE version on arXiv as a preprint. Overlap with a preprint is permitted. A preprint under your name, however, makes anonymity weaker.
2. **Review the manuscript.**
   - Read `main.pdf` in full, plus `ANALYSIS_PLAN.md` and `REPRODUCIBILITY.md`.
   - Check that you understand and agree with every claim; revise or remove any you do not.
   - Then, only if true, add to the first-page footnote in `manuscript/main.tex`: "The author reviewed all code, results and text and takes full responsibility for the content."
3. **Human-sourcing expectation.** TMLR expects ideas, claims and results to be human-sourced.
   - The research idea, hypothesis and original methodology are yours.
   - The paper's main claims come from the OpenML extension, whose design, results and interpretation came from Claude. The footnote states this.

   Decide one of the following:
   - (a) submit with full disclosure and let the editors judge;
   - (b) first rework the claims and text yourself and update the footnote to say what you did;
   - (c) do not submit to TMLR.

   Do not weaken the disclosure.
4. **Confirm the facts in the footnote and the paper:**
   - ChatGPT's role covers this revision ("research planning, writing task specifications and review").
   - You have no affiliation with NVIDIA or Kumo and no funding from them. The paper says "independent of the model's developers".
   - The "Original study (author)" sentence matches your record exactly. It is your statement, and the repository cannot confirm it.
4a. **Manuscript text you wrote yourself.** You said the initial manuscript and documentation "involved both" your work and Claude's. The 2 November IEEE disclosure and the session record show Claude drafting the text. If you wrote specific passages, name them, and the footnote will credit you. Otherwise it stays as "Claude drafted and revised the text".
4b. **Original outputs: resolved.** On 10 Nov 2026 you confirmed that no outputs, notebooks, CSV files, logs, screenshots or backups survive. Your original experiments and initial analysis are documented as your statement: not independently verified, and not described as absent. The paper reports only Claude's re-implementation and says so. Nothing was reconstructed.

**Needed for the OpenReview form (outside the PDF):**

5. Enter author details, conflicts of interest, keywords and a suggested action editor yourself. Upload `main.pdf` and the ZIP.

**Repository decisions (they don't block the upload, but they affect anonymity or licensing):**

6. **Do not push `paper/tmlr-submission`** to the public repository during review.
7. **Public repository and Kaggle notebooks.**
   - They contain this project under your name, and the ZIP's notebooks match the public Kaggle notebooks.
   - TMLR tolerates public preprints and code, but this weakens anonymity.
   - Decide whether to make the repository private during review.
8. **Font files.**
   - The 16 commercial Type 1 fonts (Times LT Std, Formata, Giovanni Std) on the public `main` have restrictive licence notices.
   - Options:
     - remove them in a new commit; history still holds them;
     - rewrite history; this needs your explicit approval;
     - obtain permission from IEEE.
9. **Remote history.** The remote history was rewritten and the IEEE branch was removed after 2 November, not by me. Confirm this was intended. A backup is in `.git-backup/`.
10. **Local commit.** Approve or amend the staging list in section 5.

**Recorded limitations (no action needed, stated in the paper):**
- untuned baselines;
- one Kumo checkpoint;
- small, numerical-only datasets;
- six regression datasets, with descriptive results only;
- related datasets;
- the `wdbc`-first analysis plan without an external timestamp;
- the `forest_fires` exclusion;
- one run on one machine;
- HistGradientBoosting differences between environments in the case study;
- AI-written verification code.

## 7. Is the package ready for TMLR submission?

**No.**
- **Technically complete:** the PDF compiles cleanly in the official style, it is anonymized, every number is traced to the data, the statistics are reproduced, and the ZIP is clean.
- **Not ready to submit until items 1–4 in section 6 are resolved:**
  - the IEEE Access conflict;
  - your review and statement of responsibility;
  - your decision on the human-sourcing expectation;
  - confirmation of the facts in the footnote.

  Only you can resolve these.
- **The contribution history is now disclosed accurately.** This removes the earlier risk of under-crediting your original work. It does not remove blockers 1–3.
- Items 4a and 4b refine the record but don't block submission.

**Checks after the second revision:**
- Clean build: 16 pages, 0 errors, 0 overfull boxes, 0 undefined references.
- 53/53 manuscript numbers match the result files.
- 0 anonymity hits in the PDF and in the ZIP.
- 21/21 result files valid; 217/217 statistics reproduced; 45/45 tests pass.
