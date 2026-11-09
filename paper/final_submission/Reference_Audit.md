# Reference Audit

Manuscript: *An Empirical Evaluation of NVIDIA Kumo Tabular Against Traditional Machine Learning Models on Small Tabular Datasets*
Audit date: 1 November 2026; re-checked 2 November 2026. Bibliography: `latex_source/references.bib`.

## Summary

| Check | Result |
|---|---|
| Entries in bibliography | 40 |
| Distinct keys cited in the text | 40 |
| Cited but missing from bibliography | none |
| In bibliography but never cited | none |
| Duplicate keys, DOIs or titles | none |
| References whose existence, title, authors, venue and year were confirmed | 40 of 40 |
| References with a detail that could not be confirmed | 2 (end page only, see below) |
| URLs checked (HTTP 200 on 1 and 2 Nov 2026) | 8 of 8 unique URLs (7 in the bibliography plus the author's ORCID URL in the title block) |
| DOIs re-checked against the DOI registry (doi.org handle API, 2 Nov 2026) | 17 of 17 registered |
| Corrections made in this audit | 3 (listed below) |

**How the checks were done.** Every entry was looked up by an automated query against an authoritative source: Crossref DOI records, arXiv API metadata (authors and published-venue comments), OpenReview, OpenAlex, the official proceedings sites (NeurIPS, PMLR, JMLR), or the original web page. The AI system that drafted the manuscript ran these checks itself; they are not an independent human verification. dblp.org and jstor.org returned bot-check pages and were not used.

**Corrections made in this audit**

1. [28] TabLLM: added the verified volume and pages (PMLR vol. 206, pp. 5549–5581), replacing the arXiv note.
2. [15] Holm: removed the issue number "2", which could not be confirmed by any reachable source. Volume 6 and pages 65–70 are confirmed (OpenAlex).
3. [39] Demšar: added the confirmed issue number (vol. 7, no. 1).

## Reference-by-reference results

Numbers follow the order of first citation, as in the PDF.

| # | Key | Reference | Verified against | Status |
|---|---|---|---|---|
| 1 | shwartzziv2022tabular | Shwartz-Ziv & Armon, *Inf. Fusion* 81:84–90, 2022 | Crossref, DOI 10.1016/j.inffus.2021.11.011 | Verified |
| 2 | grinsztajn2022tree | Grinsztajn, Oyallon & Varoquaux, NeurIPS 2022 | arXiv 2207.08815 (authors); OpenAlex (NeurIPS venue, 2022) | Verified; the "Datasets and Benchmarks Track" label is not shown in the sources checked |
| 3 | borisov2024survey | Borisov et al., *IEEE TNNLS* 35(6):7499–7519, 2024 | Crossref, DOI 10.1109/TNNLS.2022.3229161 | Verified |
| 4 | mcelfresh2023neural | McElfresh et al., NeurIPS 2023 Datasets & Benchmarks | arXiv 2305.02997 (all nine authors; comment "NeurIPS Datasets and Benchmarks Track 2023") | Verified |
| 5 | kadra2021welltuned | Kadra, Lindauer, Hutter & Grabocka, NeurIPS 2021 | arXiv 2106.11189; OpenAlex (NeurIPS) | Verified |
| 6 | hollmann2023tabpfn | Hollmann, Müller, Eggensperger & Hutter, ICLR 2023 | arXiv 2207.01848; OpenReview forum cp5PvcI6w8_ ("ICLR 2023 notable top 25%") | Verified |
| 7 | hollmann2025tabpfn | Hollmann et al., *Nature* 637(8045):319–326, 2025 | Crossref, DOI 10.1038/s41586-024-08328-6 (all eight authors) | Verified |
| 8 | qu2025tabicl | Qu, Holzmüller, Varoquaux & Le Morvan, ICML 2025 | arXiv 2502.05564 (comment "Published at ICML 2025") | Verified |
| 9 | nvidia_kumo_card | NVIDIA, Kumo Tabular model card | https://huggingface.co/nvidia/Kumo-Tabular (HTTP 200, content read) | Verified |
| 10 | nvidia_kumo_blog | Qu et al., Hugging Face community article, Sep. 2026 | https://huggingface.co/blog/nvidia/kumo-tabular (title, 13 authors and date read from the page) | Verified |
| 11 | nvidia_sdm | NVIDIA, structured-data-models 0.2.1 | GitHub repository (HTTP 200); PyPI release 0.2.1 | Verified |
| 12 | project_repo | MD. Noyon Ahmed Joy, Tabular Intelligence repository | GitHub API: public repository noyonahmedjoy/Kumo-Tabular | Verified; author name now given in full, matching the public ORCID record 0009-0004-9225-4610 |
| 13 | project_kaggle | MD. Noyon Ahmed Joy, three Kaggle notebooks | All three URLs return HTTP 200 with matching notebook titles | Verified; author name as in [12] |
| 14 | nadeau2003inference | Nadeau & Bengio, *Mach. Learn.* 52(3):239–281, 2003 | Crossref, DOI 10.1023/A:1024068626366 | Verified |
| 15 | holm1979 | Holm, *Scand. J. Statist.* 6:65–70, 1979 | OpenAlex (journal, year, volume, pages); no DOI exists | Verified after correction |
| 16 | cox1958logistic | Cox, *J. Roy. Statist. Soc. B* 20(2):215–232, 1958 | Crossref, DOI 10.1111/j.2517-6161.1958.tb00292.x | Verified |
| 17 | breiman2001rf | Breiman, *Mach. Learn.* 45(1):5–32, 2001 | Crossref, DOI 10.1023/A:1010933404324 | Verified |
| 18 | friedman2001gbm | Friedman, *Ann. Statist.* 29(5):1189–1232, 2001 | Crossref, DOI 10.1214/aos/1013203451 | Verified |
| 19 | chen2016xgboost | Chen & Guestrin, KDD 2016, pp. 785–794 | Crossref, DOI 10.1145/2939672.2939785 | Verified |
| 20 | ke2017lightgbm | Ke et al., NeurIPS 2017 | papers.nips.cc proceedings page (title and all eight authors) | Verified |
| 21 | prokhorenkova2018catboost | Prokhorenkova et al., NeurIPS 2018 | arXiv 1706.09516 (authors); papers.nips.cc ("Advances in Neural Information Processing Systems 31 (NeurIPS 2018)") | Verified |
| 22 | pedregosa2011sklearn | Pedregosa et al., *JMLR* 12:2825–2830, 2011 | jmlr.org paper page ("12(85):2825−2830") | Verified |
| 23 | arik2021tabnet | Arık & Pfister, AAAI 2021, 35(8):6679–6687 | Crossref, DOI 10.1609/aaai.v35i8.16826 | Verified |
| 24 | gorishniy2021revisiting | Gorishniy et al., NeurIPS 2021 | arXiv 2106.11959 (comment "NeurIPS 2021 camera-ready") | Verified |
| 25 | brown2020gpt3 | Brown et al., NeurIPS 2020 | arXiv 2005.14165; OpenAlex (NeurIPS 2020) | Verified |
| 26 | vaswani2017attention | Vaswani et al., NeurIPS 2017 | papers.nips.cc ("Advances in Neural Information Processing Systems 30 (NIPS 2017)") | Verified |
| 27 | garg2022icl | Garg, Tsipras, Liang & Valiant, NeurIPS 2022 | arXiv 2208.01066; OpenAlex (NeurIPS 2022) | Verified |
| 28 | hegselmann2023tabllm | Hegselmann et al., AISTATS 2023, PMLR 206:5549–5581 | proceedings.mlr.press/v206/hegselmann23a | Verified after correction |
| 29 | muller2022pfn | Müller, Hollmann, Pineda Arango, Grabocka & Hutter, ICLR 2022 | arXiv 2112.10510 (all five authors; comment "Published at ICLR 2022") | Verified |
| 30 | lee2019set | Lee et al., ICML 2019 | arXiv 1810.00825 (comment "ICML 2019") | Verified |
| 31 | dietterich1995ecoc | Dietterich & Bakiri, *JAIR* 2:263–286, 1995 | Crossref, DOI 10.1613/jair.105 | Verified |
| 32 | koenker1978quantiles | Koenker & Bassett, *Econometrica* 46(1):33–50, 1978 | Crossref, DOI 10.2307/1913643 (journal, volume, issue, first page 33) | **Partially verified: end page 50 not confirmed** |
| 33 | street1993nuclear | Street, Wolberg & Mangasarian, Proc. SPIE 1905:861–870, 1993 | Crossref, DOI 10.1117/12.148698 | Verified |
| 34 | efron2004lars | Efron, Hastie, Johnstone & Tibshirani, *Ann. Statist.* 32(2):407–499, 2004 | Crossref, DOI 10.1214/009053604000000067 | Verified |
| 35 | fawcett2006roc | Fawcett, *Pattern Recognit. Lett.* 27(8):861–874, 2006 | Crossref, DOI 10.1016/j.patrec.2005.10.010 | Verified |
| 36 | paszke2019pytorch | Paszke et al., NeurIPS 2019 | arXiv 1912.01703 (comment "NeurIPS 2019") | Verified |
| 37 | dietterich1998tests | Dietterich, *Neural Comput.* 10(7):1895–1923, 1998 | Crossref, DOI 10.1162/089976698300017197 | Verified |
| 38 | wilcoxon1945 | Wilcoxon, *Biometrics Bull.* 1(6):80–83, 1945 | Crossref, DOI 10.2307/3001968 (journal, volume, issue, first page 80) | **Partially verified: end page 83 not confirmed** |
| 39 | demsar2006comparisons | Demšar, *JMLR* 7(1):1–30, 2006 | jmlr.org; OpenAlex (vol. 7, no. 1, pp. 1–30) | Verified after correction |
| 40 | gneiting2007scoring | Gneiting & Raftery, *J. Amer. Statist. Assoc.* 102(477):359–378, 2007 | Crossref, DOI 10.1198/016214506000001437 | Verified |

## References that could not be fully verified

- **[32] Koenker & Bassett (1978)** and **[38] Wilcoxon (1945):** the papers, journals, volumes, issues and first pages are confirmed by Crossref, but the records give no end page, and JSTOR blocks automated access. Before submission, open the DOI pages (https://doi.org/10.2307/1913643 and https://doi.org/10.2307/3001968) and confirm the end pages (50 and 83).

## Citation-to-claim notes

- [10] is a community article written by the model developers. Statements attributed to it (synthetic-only pretraining, 28M to 215M parameters, benchmark rankings) are presented in the manuscript as the developers' claims, not as verified facts.
- The `small` checkpoint parameter counts (27.5M and 28.5M) were counted from the loaded models, not taken from [10].
- Architectural details of Kumo Tabular (ensemble recipe, ECOC wrapper, 20% trimmed mean) were read from the source code of structured-data-models 0.2.1 [11].
