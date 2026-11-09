# Submission Readiness Report

Target journal: IEEE Access. Last updated: 2 November 2026. The manuscript has not been submitted, peer reviewed, accepted or published. Publication on GitHub is separate from journal publication.

## Verdict: READY FOR AUTHOR SUBMISSION THROUGH IEEE SCHOLARONE

The manuscript uses the official IEEE Access template, compiles cleanly from a fresh directory, and passed visual inspection of every page. Its results were re-run identically in the original environment and reproduced at the reported precision by the author's Kaggle runs in a different environment. All 73 quoted numbers, all 40 references, all 17 DOIs and all 8 URLs were re-verified. The author has confirmed the corresponding-author e-mail, the funding status and the AI tools used, and the disclosure reflects these statements. No essential author confirmation is outstanding. Two optional items remain (see the end of this report).

## Package contents (`paper/final_submission/`)

| File | Content |
|---|---|
| `Final_IEEE_Manuscript.pdf` | Final manuscript, IEEE Access template, 10 pages, 203.2 × 276.2 mm |
| `latex_source/` | Complete editable source. In the local package it compiles standalone with pdfLaTeX. In the public repository the licence-restricted IEEE Access template files are omitted (see `latex_source/README.md`). |
| `Reference_Audit.md` | Reference-by-reference verification |
| `Reproducibility_Report.md` | Re-run, Kaggle comparison and claim-by-claim classification |
| `AI_Disclosure.md` | AI systems, evidence, author statements and the exact disclosure text |
| `FINAL_SUBMISSION_CHECKLIST.md` | Submission-day checklist |
| `reproduction/` | Re-run outputs and the Kaggle execution record |

## Status

### Author information: complete
- Name "MD. Noyon Ahmed Joy" in the title block (capitals, set by the template), the corresponding-author line, the biography, the self-citations [12] and [13], and the PDF metadata. The running header uses the ORCID family name, "Ahmed Joy".
- Affiliation "Independent Researcher, Bangladesh", as supplied by the author. No degree, employment or other credential was added.
- ORCID 0009-0004-9225-4610: valid check digit; public record listing given names "MD. Noyon" and family name "Ahmed Joy"; shown with its URL in the affiliation line and in the PDF metadata.
- Corresponding-author e-mail: the address the author confirmed in writing on 2 November 2026, shown in the template's corresponding-author line. It will be publicly visible in the PDF, on GitHub and, if the article is published, on IEEE Xplore.
- Funding: "This work did not receive any external funding." in the template's funding footnote, as confirmed by the author. No funding is recorded anywhere in the project or on the ORCID record.
- Biography: the author's public ORCID biography, used verbatim (disclosed as ChatGPT-drafted).

### AI disclosure: complete
See `AI_Disclosure.md`. The Acknowledgment names:
- **Claude Opus 5.5** (Anthropic, via Claude Code), established from the project evidence: software, local experiments, all figures and tables, statistical analysis, reference verification, and the complete manuscript text.
- **ChatGPT** (OpenAI), from the author's statement: research planning, preparing the written specifications given to Claude, review assistance, and drafting the author biography. ChatGPT did not generate results, figures, tables or analyses.

The author is credited only where evidence or the author's statement supports it: defining the research and its requirements, running the Kaggle notebooks, reviewing the project and the manuscript, and final responsibility. Codex is not named; no Codex contribution exists in the project or its history.

### Scientific and technical audit: complete
- Datasets, sample counts, splits, seeds, model and baseline settings, metrics, statistical tests (corrected resampled t-test, Holm adjustment, Wilcoxon) and hardware match the code, configuration and results (`Reproducibility_Report.md`).
- 73/73 numerical claims match their source files. Tables 4–10 regenerated from `results/` are byte-identical to the source; statistics were recomputed.
- 40 references, all cited and all verified; 17/17 DOIs registered; 8/8 unique URLs resolve. The end pages of [32] and [38] could not be confirmed from an accessible source.
- Earlier correction kept: the runtime claim states the measured ratio range (about 6 to 220 times), not a fixed order of magnitude.
- No claim is unverified. HistGradientBoosting values and runtimes are only partially verified, because they differ on Kaggle; no conclusion depends on them.

### Kaggle verification: complete
All three public notebooks show Version 1 of 1 and a successful run on a Tesla T4. Their stored outputs match the manuscript for Kumo Tabular, logistic/linear regression and random forests at the reported precision; HistGradientBoosting differs slightly. Evidence: `reproduction/kaggle_execution_evidence.md`.

### PDF validation: complete
- Clean build from an empty directory (`pdflatex`, `bibtex`, `pdflatex` ×3): no errors, no BibTeX warnings, no undefined references or citations, no missing glyphs. The only overfull-box warnings are those IEEE's unmodified sample also produces.
- 10 pages, no blank pages; all fonts embedded; no Type 3 fonts; metadata with title, author, ORCID, subject and keywords.
- All 10 pages inspected visually: figures, tables, equations, captions and references render without clipping or overlap.
- The PDF matches the source; it was compiled from the committed `latex_source/`.

### Tests and checks
| Check | Result |
|---|---|
| Unit tests | 37 passed |
| `paper/check_manuscript_numbers.py` | 73/73 |
| Citation-to-bibliography mapping | 40/40, no orphans or duplicates |
| Regenerated tables vs source | identical |
| Disclosure text in `AI_Disclosure.md` vs manuscript | identical |
| DOI registry | 17/17 |
| URLs | 8/8 |

### GitHub publication
The paper package is published on the `main` branch of https://github.com/noyonahmedjoy/Kumo-Tabular by merging `paper/ieee-access-submission`, without force-pushing or rewriting history. The submission branch is kept. The licence-restricted IEEE Access template files (commercial fonts, `ieeeaccess.cls`, `spotcolor.sty`, logos, template zip) are excluded via `.gitignore`. The staged files contain no secrets, credentials or local paths; the only e-mail address is the author-approved corresponding-author address.

## Remaining optional items

| Item | Status |
|---|---|
| Confirm the end pages of references [32] (doi:10.2307/1913643) and [38] (doi:10.2307/3001968) on the publisher pages | Recommended; detail level only |
| Biography photograph | Optional; the template supports one |

## Steps the author takes in IEEE ScholarOne

1. Link ORCID 0009-0004-9225-4610 to the ScholarOne account.
2. Upload `paper/final_submission/Final_IEEE_Manuscript.pdf` and the LaTeX source from the **local** `paper/final_submission/latex_source/` (which includes the template files), with `references.bib` and `main.bbl`.
3. Answer the AI-use question in the submission form consistently with the Acknowledgment.
4. Do not describe the article as accepted or published until IEEE completes peer review.
