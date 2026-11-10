# Reference audit: TMLR manuscript

Bibliography: `paper/tmlr_submission/manuscript/references.bib`. Style: `tmlr.bst` (natbib, author–year). Audit date: 9 November 2026.

## Summary

| Check | Result |
|---|---|
| Entries in the bibliography | 44 |
| Entries carried over unchanged from the IEEE bibliography (`paper/final_submission/latex_source/references.bib`) | 37 |
| Carried-over entries changed | 1 (`nvidia_kumo_card`: `year = {2026}` added, see below) |
| IEEE entries removed | 2 (`project_repo`, `project_kaggle`: self-references, replaced by an anonymized citation) |
| New entries | 6 (5 scientific references, plus 1 anonymized supplementary-material entry); all 5 scientific entries verified against an authoritative source |
| Distinct keys cited in the text | 44 of 44 (`main.bbl` has 44 items) |
| Cited but missing from the bibliography | none (no undefined citations in `main.log`) |
| In the bibliography but never cited | none (`fawcett2006roc` and `paszke2019pytorch` were uncited; both are now cited where ROC-AUC and PyTorch are used) |
| BibTeX warnings | 1: `arik2021tabnet` has both volume and number; `tmlr.bst` prints the volume only. Both values are correct (Crossref: vol. 35, no. 8) |

## Carried-over references

The 38 entries carried over from the IEEE bibliography (37 unchanged plus `nvidia_kumo_card`) were verified in the IEEE reference audit (`paper/final_submission/Reference_Audit.md`, 1–2 November 2026) against Crossref, arXiv, OpenReview, OpenAlex, official proceedings sites or the original web page. That audit lists the method and result for each entry, including two entries where only the end page could not be confirmed. The entries were not re-verified one by one for this manuscript. The TMLR style renders them in author–year form, and their bibliographic content is unchanged.

`nvidia_kumo_card`: the IEEE entry had no year, and author–year style needs one. The year 2026 is the creation date of the Hugging Face repository `nvidia/Kumo-Tabular` (repository metadata `createdAt`, 2026-09-01).

## New references

| Key | Cited for | Verification |
|---|---|---|
| `vanschoren2014openml` | OpenML platform | Crossref, DOI 10.1145/2641190.2641198: title, authors (Vanschoren, van Rijn, Bischl, Torgo), vol. 15, no. 2, pp. 49–60, 2014. Re-checked 9 Nov 2026 |
| `bischl2021openml` | OpenML-CC18 benchmark suite | arXiv 1708.03731: title "OpenML Benchmarking Suites"; journal reference "NeurIPS 2021 Datasets and Benchmarks Track"; all nine authors match, in order. Re-checked 10 Nov 2026 |
| `fischer2023ctr23` | OpenML-CTR23 regression suite | **Verified on 10 Nov 2026 against the official OpenML study page** (`https://www.openml.org/api/v1/json/study/353`). That page carries the authors' own citation for the suite: title, booktitle "AutoML Conference 2023 (Workshop)", year 2023, URL `openreview.net/forum?id=HebAOoMm94`. **Correction:** the entry was missing the second author, Liana Harutyunyan, who is now added. The earlier pass on 9 Nov had wrongly marked this entry as verified. The authors' BibTeX on that page lacks a separator between two author names; the four authors are Fischer, Harutyunyan, Feurer and Bischl. OpenReview, Semantic Scholar and dblp could not be used (bot-check pages or rate limits, not bypassed). |
| `friedman1937ranks` | Friedman test | Crossref, DOI 10.1080/01621459.1937.10503522: vol. 32, no. 200, pp. 675–701, 1937. Re-checked 9 Nov 2026 |
| `geurts2006extremely` | Extra trees baseline | Crossref, DOI 10.1007/s10994-006-6226-1: Geurts, Ernst, Wehenkel; vol. 63, no. 1, pp. 3–42, 2006. Re-checked 9 Nov 2026 |
| `anonymous2026supplement` | Code, notebooks and raw results | Not a publication. It is an anonymized pointer to the supplementary ZIP, used where the IEEE version cited the author's GitHub repository and Kaggle notebooks |

## Self-references and anonymization

The IEEE manuscript cited the author's public GitHub repository and Kaggle notebooks (`project_repo`, `project_kaggle`). Both would identify the author. They were not deleted without replacement. Each place that cited them now cites `anonymous2026supplement`:
- the Reproducibility Statement in `main.tex`;
- the sentence on the cloud re-execution of the notebooks in `sections/design.tex`.

The content they pointed to is in the supplementary ZIP.

The IEEE article itself is not cited, because it has not been published. If it is published or under review elsewhere, TMLR's policy on prior and concurrent submissions applies. **This is a decision for the author** (see the final report).

No scientific reference was removed for anonymity.

## Limits of this audit

The verification was done by the AI assistant using automated queries, not by an independent human. dblp, JSTOR and OpenReview returned bot-check pages at various points; these were not bypassed.
