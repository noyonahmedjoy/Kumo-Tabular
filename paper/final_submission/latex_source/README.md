# LaTeX source (IEEE Access template)

Editable source of `../Final_IEEE_Manuscript.pdf`, built on the official IEEE Access LaTeX template (`ACCESS_latex_template_20260513`, from https://ieeeaccess.ieee.org/authors/preparing-your-article/; a copy of the zip is in `paper/`).

| Path | Content |
|---|---|
| `main.tex` | manuscript |
| `references.bib` | bibliography (40 entries, all cited) |
| `main.bbl` | BibTeX output, for submission systems that do not run BibTeX |
| `tables/` | six results tables, generated from `results/` |
| `figures/` | figures used by the manuscript; `paradigms_diagram.tex` is the source of Fig. 1 |
| `ieeeaccess.cls`, `IEEEtran.cls`, `IEEEtran.bst`, `spotcolor.sty` | template class, bibliography style and support package, unmodified |
| `logo.png`, `notaglinelogo.png`, `bullet.png` | template graphics, unmodified |
| `t1-*.pfb`, `t1-*.tfm`, `t1*.fd`, `t1-*.map` | template fonts (Times, Formata, Giovanni), unmodified |

The local submission package is self-contained. In the public GitHub repository, `ieeeaccess.cls`, `spotcolor.sty`, the three template PNG files and the `t1*` font files are excluded via `.gitignore`: the fonts are commercially licensed (Adobe, "All Rights Reserved") and the class has no licence statement. `IEEEtran.cls` and `IEEEtran.bst` are distributed under the LaTeX Project Public License and are included. To build from a clone, copy the excluded files from the official template zip into this folder.

## Build

The IEEE Access class requires **pdfLaTeX**. It loads its fonts with `\pdfmapfile`, and the template's own sample fails to compile under XeTeX (tested; LuaTeX was not tested).

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Tested with pdfTeX 1.40.29 (TeX Live 2026, TinyTeX v2026.10); the same toolchain reproduces the template's sample PDF. Overleaf with the pdfLaTeX compiler should also work.

The build log contains two warnings that also appear when compiling IEEE's unmodified sample: a 9.27 pt overfull box from the title block and repeated 505 pt overfull boxes during page output. They come from the template, not from the manuscript content.

## Fig. 1

`ieeeaccess.cls` conflicts with TikZ when TikZ is loaded in the manuscript preamble, so the diagram is a standalone document compiled to `figures/paradigms_diagram.pdf`:

```bash
cd figures
pdflatex paradigms_diagram
```

It refers to Eq. (1) and Eq. (2) by number. If equations are renumbered, update the diagram.

## Regenerating tables and figures

From the repository root, with the project environment installed:

```bash
python paper/build_paper_assets.py
python paper/check_manuscript_numbers.py
```

The first command rebuilds `tables/` and the data figures from `results/`. To redraw the precision-recall figure it re-runs the seed-42 classification split, and it stops if any recomputed metric differs from `results/`. The second command checks 73 numbers quoted in the text of `main.tex` against their source files.
