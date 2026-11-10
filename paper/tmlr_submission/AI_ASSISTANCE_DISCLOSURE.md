# AI Assistance Disclosure and Contribution Record (TMLR submission)

Prepared 9 November 2026. Revised 10 November 2026 using the author's confirmed contribution history and checked against the Claude Code session record. This file is not part of the anonymized submission. It documents the basis for the first-page footnote in the manuscript.

## TMLR policy (re-read 10 November 2026)

- **Editorial Policies, "LLM use policy for authors":**
  - LLMs may be used as general-purpose assistive tools.
  - Authors are fully responsible for content on which they are listed as authors, including LLM-generated content that could be construed as plagiarism or misconduct, such as fabrication of facts.
  - Submissions that appear to be largely LLM-generated "will be closely examined".
  - LLMs are not eligible for authorship.
- **FAQ:** TMLR expects that "the ideas, claims and results found in submissions are human-sourced". It requires that authors mention LLM use "as a footnote on the first page".

**How the manuscript complies:**
- The disclosure is a footnote on page 1.
- It names the tools, Claude Opus 5.5 (via Claude Code) and ChatGPT.
- It separates the author's original work from the AI-assisted work.
- It lists no AI system as an author.

**What compliance does not settle:**
- **Human sourcing:** the expectation is met for the original idea, hypothesis and methodology. It is not met for the design, results and interpretation of the OpenML extension, which are the paper's main claims (see Action 2).
- **Responsibility:** the author must still accept responsibility for the content (see Action 1).

## Contribution record

### A. Original two-dataset study: the author's own work

The author confirmed on 10 November 2026 that these contributions were completed independently, before Claude's involvement. No dates are recorded, and none are stated. The author also confirmed that **no outputs, notebooks, CSV files, logs, screenshots or backups** of the original experiments survive.

In the "Source" column below:
- **Author statement** means the item rests on the author's word alone. It is not independently verified, and no evidence contradicts it.
- **Author statement + repository evidence** means the author's word, partly supported by files, Git history or the session record.

| Contribution | Source |
|---|---|
| Originated the research idea and hypothesis | **Author statement + repository evidence.** The idea is set out in the author's brief of 31 Oct 2026. Its earlier, independent origin is the author's statement |
| Developed the initial benchmark methodology | **Author statement + repository evidence.** The written brief given to Claude on 31 Oct 2026 contains this methodology: datasets, train/test separation, context and query rows, never exposing test labels to Kumo, handling of regression quantiles |
| Selected the evaluation metrics and baseline models | **Author statement + repository evidence.** Listed in the 31 Oct brief: accuracy, precision, recall, F1, ROC-AUC and confusion matrix; MAE, RMSE and R²; logistic or linear regression, random forest and histogram gradient boosting |
| Ran experiments on the two original datasets | **Author statement** (10 Nov 2026). No surviving outputs (author's confirmation; none found in the repository, `audit/PROVENANCE_AUDIT.md`). Not independently verified. No result from these runs is reported or reconstructed in the manuscript |
| Performed the initial analysis of the original experimental results | **Author statement** (10 Nov 2026). No surviving artifact (author's confirmation). Not independently verified |

### B. Work done with Claude in this repository (session record, 31 October to 10 November 2026)

**Two-dataset study, as reported in the manuscript (case studies):**

| Contribution | Who | Source |
|---|---|---|
| Re-implementation from the author's specification: code, notebooks, tests | Claude Opus 5.5 (Anthropic), via Claude Code | Session history; `src/`, `scripts/`, `notebooks/` |
| Execution of the local runs that produced the reported two-dataset numbers | Claude | `results/` |
| Additions beyond the brief: five repeated splits, training-size study, evaluation of predictive intervals | Claude, 31 Oct 2026 | Session history. None of these appears in the brief |
| Corrected significance tests (Nadeau–Bengio corrected t-test, Holm correction) | Claude, 1 Nov 2026, in response to the author's instruction not to claim significance without evidence | Session history |
| Execution of the Claude-generated notebooks on Kaggle (Tesla T4); results match the local run except HistGradientBoosting | Author. This reproduces Claude's re-implementation, not the original experiments | `paper/final_submission/reproduction/kaggle_execution_evidence.md` |

**OpenML extension:**

| Contribution | Who | Source |
|---|---|---|
| Decision to extend with OpenML datasets; choice of XGBoost, LightGBM and CatBoost | Author, choosing among options Claude proposed. Claude had marked the OpenML option "Recommended" | Session history, 9 Nov 2026 |
| Design: inclusion criteria (22 datasets selected, 21 evaluated), baseline configurations, statistical analysis plan | Claude. The author did not review the plan before it was used | `ANALYSIS_PLAN.md` |
| Code, execution, validation, statistical analysis and its verification, figures and tables | Claude | `scripts/`, `results/openml_benchmark/`, `paper/tmlr_submission/*.py` |
| Interpretation of results, discussion, conclusions, limitations | Claude | Session history |
| Handling of deviations (`forest_fires` exclusion, runner change) | Claude, applying the plan's criterion. Reported to the author afterwards | `ANALYSIS_PLAN.md` |
| Reference search and verification | Claude | `audit/REFERENCE_AUDIT.md` |
| TMLR manuscript text, LaTeX, anonymization, supplementary archive | Claude, drafting and revising on the author's instructions | Session history |

### C. Manuscripts and documentation

- **The author's statement of 10 November 2026:** the initial manuscript, research documentation and methodology documentation "involved both my own work and Claude's assistance".
- **The session record:** the IEEE Access text in this repository was drafted by Claude. The IEEE disclosure the author approved on 2 November 2026 says Claude "drafted the complete text of this article" and that the author reviewed it.
- **The TMLR manuscript text** was drafted and revised by Claude, following the author's instructions.
- The footnote therefore says Claude drafted and revised the text. **If the author wrote particular passages themselves, the author should name them so the footnote can say so.**

### D. ChatGPT

ChatGPT (OpenAI; model version not recorded) was used for research planning, preparing the written specifications and review assistance, according to the author's statement of 2 November 2026. Whether it was also used for the TMLR-phase instructions is not confirmed.

## Provenance of the reported results

`audit/PROVENANCE_AUDIT.md` traces every case-study number in the manuscript to Claude's local run of 31 October 2026 (`results/`, unchanged since the first commit). No output of the author's original experiments exists in the repository or its Git history.

The manuscript therefore:
- attributes the reported two-dataset results to the re-implementation;
- says, in Section 4, that the outputs of the earlier experiments are not available;
- makes no claim that they are reproduced.

## Footnote used in the manuscript (page 1)

> **Use of large language models and division of work.** *Original study (author).* Before Claude's involvement, the author independently originated the research idea and hypothesis, developed the initial benchmark methodology, selected the evaluation metrics and baseline models, ran experiments on the two original datasets and performed an initial analysis of their results. *AI-assisted work.* Claude Opus 5.5 (Anthropic), used through Claude Code, re-implemented the two-dataset study from the author's specification and produced the two-dataset results reported here (Section 5.5, Appendix B), adding the repeated splits, the training-size study, the interval evaluation and the corrected significance tests. For the OpenML benchmark, the author chose to extend the study with OpenML datasets and chose the additional baseline libraries from options Claude proposed; Claude designed the extension (dataset inclusion criteria, baseline configurations, statistical analysis plan), wrote all code, ran the experiments, performed and interpreted the statistical analysis and checked the references. Claude drafted and revised the text, tables and figures, including the conclusions, following the author's instructions. ChatGPT (OpenAI) assisted the author with research planning, writing task specifications and review.

**Why each point is worded as it is:**
- **The author's original work** is stated in full, as the author confirmed it, and is not shortened.
- **"Re-implemented … and produced the two-dataset results reported here."** The numbers in the paper come from Claude's code and Claude's runs, not from the author's earlier runs. Without this sentence, the case-study results would wrongly read as the author's own experiments.
- **The OpenML extension** is attributed to Claude, apart from the two choices the author made.
- **Responsibility:** the footnote does not say the author reviewed the paper or takes responsibility, because the author has not yet reviewed this version (Action 1).
- **Anonymity:** the footnote contains no identifying information.

## Actions required from the author

1. **Review this version.** Read the manuscript, the analysis plan and the results. Only if true, add to the footnote: "The author reviewed all code, results and text and takes full responsibility for the content."
2. **Human sourcing of the main claims.** The paper's main claims come from the OpenML extension, whose design, results and interpretation came from Claude. Under TMLR's expectation, decide one of the following:
   - (a) submit with this full disclosure and let the editors judge;
   - (b) first verify and rework the claims and text yourself, then update the footnote to describe that work;
   - (c) do not submit to TMLR.
3. **Original outputs: resolved, 10 Nov 2026.** The author confirmed that none survive. The original experiments stay documented as the author's statement. No action is needed.
4. **Manuscript authorship detail.** If you wrote specific text yourself, in the IEEE draft or the documentation, name it so it can be attributed to you. Otherwise the footnote stays as it is.
5. **ChatGPT.** Confirm that its role in the TMLR phase is fully described.
6. **Independence.** Confirm that you have no affiliation with NVIDIA or Kumo, and no funding from them. The paper calls the evaluation independent of the developers.
