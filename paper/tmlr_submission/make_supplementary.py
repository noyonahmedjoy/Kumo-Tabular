import re
import sys
import zipfile
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "supplementary" / "anonymized_code_and_results.zip"
INCLUDE = ["src", "scripts", "tests", "configs", "notebooks", "results", "requirements.txt", "data/README.md",
           "paper/tmlr_submission/audit/ANALYSIS_PLAN.md", "paper/tmlr_submission/verify_statistics.py"]
EXCLUDE_PARTS = {"__pycache__", ".ipynb_checkpoints"}
EXCLUDE_NAMES = {"console_output.txt", "console_output_resume.txt", "build_notebooks.py"}
IDENTIFYING = re.compile(
    r"noyon|nayon|ahmed|joyml|imtiaz|orcid|0009-0004|C:[\\/]+Users|F:[\\/]+project|github\.com/noyon|kaggle\.com/code/nayon"
    r"|[\w.+-]+@[\w-]+\.(?:com|org|net|edu|io)\b",
    re.IGNORECASE)
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".ipynb", ".tex", ".cfg", ".toml"}
README = """# Supplementary material: code and results

Anonymized code and raw results for the submission "An Empirical Evaluation of a Pretrained Tabular Foundation Model
on Small Numerical Tables". No author-identifying information is included.

## Contents

| Path | Content |
|---|---|
| `src/` | Data loading (scikit-learn and OpenML), Kumo Tabular wrapper, baselines, metrics, benchmark runners, plots |
| `scripts/run_all.py` | Original two-dataset study (breast cancer, diabetes) |
| `scripts/select_openml_suite.py` | Applies the dataset inclusion criteria to OpenML-CC18 and OpenML-CTR23 |
| `scripts/run_openml_benchmark.py` | Runs the OpenML benchmark (21 datasets) on the official OpenML 10-fold splits |
| `scripts/validate_openml_results.py` | Checks that every result file is complete and every metric valid |
| `scripts/analyze_openml_benchmark.py` | Pre-specified cross-dataset analysis (Friedman, Holm-corrected Wilcoxon, sensitivity analyses) |
| `paper/tmlr_submission/audit/ANALYSIS_PLAN.md` | Analysis plan written before the results were inspected, with recorded deviations |
| `paper/tmlr_submission/verify_statistics.py` | Independent recomputation of all reported statistics from the raw per-fold results |
| `tests/` | Unit tests, including data-leakage and prediction-shape checks |
| `configs/` | Experiment configuration and the selected OpenML suite |
| `results/` | Raw per-fold and per-split results, run logs, environment records, analysis outputs and figures |
| `notebooks/` | Self-contained notebooks for the original two-dataset study |

## Reproduce

Python 3.12, PyTorch with CUDA support for your GPU (CPU also works, more slowly), then:

```bash
pip install -r requirements.txt xgboost lightgbm catboost
python -m pytest
python scripts/run_all.py
python scripts/run_openml_benchmark.py
python scripts/validate_openml_results.py
python scripts/analyze_openml_benchmark.py
python paper/tmlr_submission/verify_statistics.py
```

The Kumo Tabular `small` checkpoints are downloaded from Hugging Face (`nvidia/Kumo-Tabular`) on first use; the OpenML
datasets and splits are downloaded from openml.org. Exact package versions of the reported run are in
`results/openml_benchmark/environment.json`. The suite selected by `scripts/select_openml_suite.py` contains 22
datasets; `forest_fires` (OpenML 44962) is excluded at load time because it contains text columns that its OpenML
metadata does not report (`configs/openml_suite.json`, `excluded_at_load`).
"""


def without_metadata(path: Path) -> bytes:
    document = pymupdf.open(path)
    document.set_metadata({})
    document.del_xml_metadata()
    return document.tobytes(garbage=3, deflate=True)


def collect() -> list[Path]:
    files = []
    for item in INCLUDE:
        path = ROOT / item
        candidates = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for file in candidates:
            if EXCLUDE_PARTS & set(file.relative_to(ROOT).parts) or file.name in EXCLUDE_NAMES:
                continue
            files.append(file)
    return files


def main() -> None:
    files = collect()
    leaks = []
    for file in files:
        if file.suffix in TEXT_SUFFIXES:
            text = file.read_text(encoding="utf-8", errors="replace")
            for match in IDENTIFYING.finditer(text):
                leaks.append(f"{file.relative_to(ROOT)}: {match.group(0)}")
    if IDENTIFYING.search(README):
        leaks.append("README: identifying text")
    if leaks:
        print("Identifying information found; archive not written:")
        print("\n".join(leaks[:50]))
        sys.exit(1)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.md", README)
        for file in files:
            info = zipfile.ZipInfo(file.relative_to(ROOT).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, without_metadata(file) if file.suffix == ".pdf" else file.read_bytes())
    size = OUTPUT.stat().st_size / 1024 / 1024
    print(f"Wrote {OUTPUT.relative_to(ROOT)}: {len(files) + 1} files, {size:.2f} MB")


if __name__ == "__main__":
    main()
