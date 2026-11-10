import re
import sys
import zipfile
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
PATTERN = re.compile(
    r"noyon|nayon|ahmed|joyml|imtiaz|orcid|0009-0004|github\.com/noyon|kaggle\.com/code|nayonahmedlol"
    r"|C:[\\/]+Users|F:[\\/]+project|independent researcher|bangladesh"
    r"|[\w.+-]+@[\w-]+\.(?:com|org|net|edu|io)\b",
    re.IGNORECASE)


def scan_pdf(path: Path) -> list[str]:
    document = pymupdf.open(path)
    hits = []
    for number, page in enumerate(document, 1):
        hits += [f"page {number}: {m.group(0)}" for m in PATTERN.finditer(page.get_text())]
        for link in page.get_links():
            uri = link.get("uri") or ""
            hits += [f"page {number} link: {uri}" for _ in PATTERN.finditer(uri)]
    for key, value in document.metadata.items():
        if value and PATTERN.search(str(value)):
            hits.append(f"metadata {key}: {value}")
        if key in ("creationDate", "modDate") and value:
            hits.append(f"metadata {key} reveals a time zone: {value}")
    xmp = document.get_xml_metadata() or ""
    hits += [f"XMP: {m.group(0)}" for m in PATTERN.finditer(xmp)]
    return hits


def scan_zip(path: Path) -> list[str]:
    hits = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if PATTERN.search(info.filename):
                hits.append(f"file name: {info.filename}")
            data = archive.read(info).decode("utf-8", errors="replace")
            hits += [f"{info.filename}: {m.group(0)}" for m in PATTERN.finditer(data)]
            if info.comment:
                hits.append(f"{info.filename}: has a comment")
            if info.filename.endswith(".pdf"):
                metadata = pymupdf.open(stream=archive.read(info), filetype="pdf").metadata
                hits += [f"{info.filename}: metadata {k} = {v}" for k, v in metadata.items()
                         if k in ("creationDate", "modDate") and v]
        if archive.comment:
            hits.append("archive comment present")
    return hits


def main() -> None:
    targets = [Path(p) for p in sys.argv[1:]] or [HERE / "manuscript" / "main.pdf"]
    total = 0
    for target in targets:
        hits = scan_zip(target) if target.suffix == ".zip" else scan_pdf(target)
        total += len(hits)
        print(f"{target.name}: {len(hits)} identifying hits")
        for hit in hits[:40]:
            print("   ", hit)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
