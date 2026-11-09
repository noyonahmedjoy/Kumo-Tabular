# Final Submission Checklist

## Author decisions

- [x] Corresponding-author e-mail confirmed and added (2 Nov 2026)
- [x] Funding statement: no external funding (2 Nov 2026)
- [x] AI disclosure: Claude and ChatGPT named, with their contributions and the affected content (2 Nov 2026)
- [x] Paper package published on GitHub `main` by merging `paper/ieee-access-submission`
- [ ] Optional: confirm the end pages of references [32] (doi:10.2307/1913643) and [38] (doi:10.2307/3001968)
- [ ] Optional: add a biography photo (`IEEEbiography` with `\includegraphics`, 1 × 1.25 in)

## Rebuild after any edit

- [ ] `cd paper/final_submission/latex_source` (local package, which contains the template files)
- [ ] `pdflatex main`, `bibtex main`, `pdflatex main`, `pdflatex main`
- [ ] The log has no errors, no "undefined", no "Rerun"; only the template's own 9.27 pt / 505 pt overfull boxes
- [ ] From the repository root: `python paper/check_manuscript_numbers.py` reports 73/73
- [ ] Copy `main.pdf` to `../Final_IEEE_Manuscript.pdf` and look through every page

## Verified on 2 November 2026

- [x] IEEE Access template class, front matter, alphabetical index terms, 248-word abstract
- [x] Author name, affiliation, ORCID, e-mail and funding consistent in the title block, biography, references and PDF metadata
- [x] Clean build from an empty directory; fonts embedded; no Type 3 fonts; no missing glyphs; no blank pages
- [x] All 10 pages inspected
- [x] 73/73 quoted numbers; 40/40 references; 17/17 DOIs; 8/8 URLs
- [x] Results re-run identically locally; reproduced on Kaggle (T4) at the reported precision except HistGradientBoosting
- [x] Unit tests 37/37

## At submission (IEEE Access, ScholarOne)

- [ ] Link ORCID 0009-0004-9225-4610 to the ScholarOne account
- [ ] Upload `paper/final_submission/Final_IEEE_Manuscript.pdf` and the local `latex_source/` (including `references.bib`, `main.bbl` and the template files)
- [ ] Answer the AI-use question consistently with the Acknowledgment
- [ ] Do not describe the article as accepted or published until IEEE completes peer review
