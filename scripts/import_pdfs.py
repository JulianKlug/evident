"""List manifest PDFs still missing; import hand-downloaded PDFs under their manifest filename.

    missing:  manifest ──▶ no valid PDF in the PDF dir? ──▶ https://doi.org/<doi> links
              --open: all links opened as tabs in Chrome (macOS `open`), ready for "Download PDF"

    import:   ~/Downloads/*.pdf ──DOI from PDF metadata or first pages──▶ manifest entry
                  │ exactly one manifest DOI          │ none / several
                  ▼                                   ▼
              copy to <pdf dir>/<pdf_filename>     reported, never copied

Usage:
    python scripts/import_pdfs.py missing [--open] [--manifest corpus/manifest.csv] [--pdf-dir DIR]
    python scripts/import_pdfs.py import ~/Downloads [--manifest ...] [--pdf-dir DIR]

Never imports extraction.pdf_loader (its DOI fallback reaches Sci-Hub).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypdf import PdfReader  # noqa: E402
from pypdf.errors import PdfReadError  # noqa: E402

from evident.corpus import ManifestEntry, load_manifest  # noqa: E402
from utils.doi import normalize_doi  # noqa: E402

_DEFAULT_MANIFEST = "corpus/manifest.csv"
_DEFAULT_PDF_DIR = "/mnt/data1/klug/datasets/evidence_extraction/pdfs"
_DOI_RESOLVER = "https://doi.org/{doi}"
_PDF_MAGIC = b"%PDF"
_PDF_SUFFIX = ".pdf"
_PAGES_SCANNED = 2  # the article DOI is on the first page; page 2 covers cover sheets
_DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s\"<>,;]+")
_TRAILING_PUNCTUATION = ".)]}"
_EXIT_OK = 0
_EXIT_ERROR = 1
_MACOS = "darwin"
_BROWSER_APP = "Google Chrome"


class LinkMode(str, Enum):
    PRINT = "print"  # list the links
    OPEN = "open"    # also open them as browser tabs


class ImportStatus(str, Enum):
    IMPORTED = "imported"
    ALREADY_PRESENT = "already_present"  # the manifest file exists and is a valid PDF; left untouched
    UNMATCHED = "unmatched"              # no manifest DOI found in the PDF
    AMBIGUOUS = "ambiguous"              # several manifest DOIs found; which one is unclear
    UNREADABLE = "unreadable"            # not a PDF, or pypdf can't parse it


@dataclass(frozen=True)
class ImportResult:
    source: str
    status: ImportStatus
    doi: str = ""
    detail: str = ""


def missing(entries: list[ManifestEntry], pdf_dir: str) -> list[ManifestEntry]:
    """Manifest entries without a valid PDF on disk, in manifest order."""
    return [e for e in entries if not _is_pdf(os.path.join(pdf_dir, e.pdf_filename))]


def open_in_browser(urls: list[str]) -> None:
    """One `open -a "Google Chrome" url…` call: every link becomes a tab. macOS only."""
    if sys.platform != _MACOS:
        raise OSError(f"--open uses the macOS `open` command; on {sys.platform}, open the printed links by hand")
    subprocess.run(["open", "-a", _BROWSER_APP, *urls], check=True)


def import_dir(source_dir: str, entries: list[ManifestEntry], pdf_dir: str) -> list[ImportResult]:
    by_doi = {e.doi: e for e in entries}
    os.makedirs(pdf_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(source_dir) if n.lower().endswith(_PDF_SUFFIX))
    return [_import_one(os.path.join(source_dir, n), by_doi, pdf_dir) for n in names]


def dois_in(path: str) -> set[str]:
    """Normalized DOIs from the PDF metadata and the text of its first pages."""
    reader = PdfReader(path)
    texts = [str(v) for v in (reader.metadata or {}).values()]
    texts += [page.extract_text() or "" for page in reader.pages[:_PAGES_SCANNED]]
    return {normalize_doi(m.rstrip(_TRAILING_PUNCTUATION)) for t in texts for m in _DOI_PATTERN.findall(t)}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="List missing manifest PDFs; import hand-downloaded ones.")
    parser.add_argument("--manifest", default=_DEFAULT_MANIFEST)
    parser.add_argument("--pdf-dir", default=_DEFAULT_PDF_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    miss = commands.add_parser("missing", help="print a DOI link per manifest guideline without a PDF")
    miss.add_argument("--open", action="store_true", help=f"also open every link in {_BROWSER_APP} (macOS)")
    imp = commands.add_parser("import", help="copy downloaded PDFs to the PDF dir, named by manifest DOI")
    imp.add_argument("source_dir")
    args = parser.parse_args(argv)

    entries = load_manifest(args.manifest)
    if args.command == "missing":
        return _report_missing(entries, args.pdf_dir, LinkMode.OPEN if args.open else LinkMode.PRINT)

    results = import_dir(args.source_dir, entries, args.pdf_dir)
    for r in results:
        if r.status != ImportStatus.IMPORTED:
            print(f"{r.status.value:16s} {os.path.basename(r.source)}  {r.doi} {r.detail}".rstrip())
    for status in ImportStatus:
        print(f"{status.value:16s} {sum(r.status == status for r in results)}")
    print(f"still missing: {len(missing(entries, args.pdf_dir))} of {len(entries)}")
    return _EXIT_OK


def _report_missing(entries: list[ManifestEntry], pdf_dir: str, mode: LinkMode) -> int:
    gaps = missing(entries, pdf_dir)
    for e in gaps:
        print(f"{_DOI_RESOLVER.format(doi=e.doi)}  {e.society} {e.year}  {e.title[:70]}")
    print(f"{len(gaps)} of {len(entries)} missing")
    if mode == LinkMode.PRINT or not gaps:
        return _EXIT_OK

    try:
        open_in_browser([_DOI_RESOLVER.format(doi=e.doi) for e in gaps])
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"error: {e}", file=sys.stderr)
        return _EXIT_ERROR
    print(f"opened {len(gaps)} tabs in {_BROWSER_APP}; download each PDF, then: import ~/Downloads")
    return _EXIT_OK


def _import_one(path: str, by_doi: dict[str, ManifestEntry], pdf_dir: str) -> ImportResult:
    if not _is_pdf(path):
        return ImportResult(path, ImportStatus.UNREADABLE, detail="not a PDF")
    try:
        found = dois_in(path)
    except (PdfReadError, ValueError, OSError) as e:
        return ImportResult(path, ImportStatus.UNREADABLE, detail=str(e))

    # Exactly one manifest DOI, or nothing is copied
    matches = sorted(d for d in found if d in by_doi)
    if not matches:
        return ImportResult(path, ImportStatus.UNMATCHED, detail=f"DOIs found: {', '.join(sorted(found)) or 'none'}")
    if len(matches) > 1:
        return ImportResult(path, ImportStatus.AMBIGUOUS, detail=", ".join(matches))

    doi = matches[0]
    dest = os.path.join(pdf_dir, by_doi[doi].pdf_filename)
    if _is_pdf(dest):
        return ImportResult(path, ImportStatus.ALREADY_PRESENT, doi)
    _copy_atomic(path, dest)
    return ImportResult(path, ImportStatus.IMPORTED, doi)


def _is_pdf(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    with open(path, "rb") as f:
        return f.read(len(_PDF_MAGIC)) == _PDF_MAGIC


def _copy_atomic(src: str, dest: str) -> None:
    """Copy to <dest>.part, then rename: a crash never leaves a half PDF under the manifest name."""
    part = dest + ".part"
    try:
        shutil.copyfile(src, part)
        os.replace(part, dest)
    finally:
        if os.path.exists(part):
            os.remove(part)


if __name__ == "__main__":
    sys.exit(main())
