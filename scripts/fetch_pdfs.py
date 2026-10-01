"""Fetch the PDF of every manifest guideline into the PDF dir; write a report of what is missing.

    manifest entry ──▶ already on disk? ──yes──▶ PRESENT
                           │ no
                           ▼
                      Unpaywall best_oa_location.url_for_pdf ──PDF──▶ DOWNLOADED
                           │ none / refused
                           ▼
                      https://doi.org/<doi> → <meta citation_pdf_url> ──PDF──▶ DOWNLOADED
                           │ none / refused
                           ▼
                      PAYWALLED | INVALID_PDF | NOT_FOUND

Usage (inside the institutional VPN for paywalled journals):
    python scripts/fetch_pdfs.py --email YOU@ORG [--manifest corpus/manifest.csv] [--pdf-dir DIR]

Never imports extraction.pdf_loader (its DOI fallback reaches Sci-Hub).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, fields
from enum import Enum
from html.parser import HTMLParser
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evident.corpus import ManifestEntry, load_manifest  # noqa: E402

_DEFAULT_MANIFEST = "corpus/manifest.csv"
_DEFAULT_PDF_DIR = "/mnt/data1/klug/datasets/evidence_extraction/pdfs"
_DEFAULT_REPORT = "fetch_report.csv"
_UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}?email={email}"
_DOI_RESOLVER = "https://doi.org/{doi}"
_PDF_MAGIC = b"%PDF"
_PDF_CONTENT_TYPE = "application/pdf"
_HTTP_OK = 200
_HTTP_REFUSED = {401, 403}
_REQUEST_DELAY_S = 1.0
_TIMEOUT_S = 60
_USER_AGENT = "evident-fetch/1.0 (living evidence map; research use)"
_PART_SUFFIX = ".part"


class FetchStatus(str, Enum):
    PRESENT = "present"
    DOWNLOADED = "downloaded"
    PAYWALLED = "paywalled"
    NOT_FOUND = "not_found"
    INVALID_PDF = "invalid_pdf"


class _Source(str, Enum):
    DISK = "disk"
    UNPAYWALL = "unpaywall"
    LANDING_PAGE = "landing_page"


@dataclass(frozen=True)
class FetchResult:
    doi: str
    status: FetchStatus
    source: str = ""
    sha256: str = ""
    detail: str = ""


def fetch_all(entries: list[ManifestEntry], pdf_dir: str, email: str) -> list[FetchResult]:
    os.makedirs(pdf_dir, exist_ok=True)
    return [_fetch_one(e, pdf_dir, email) for e in entries]


def write_report(results: list[FetchResult], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(FetchResult)])
        writer.writeheader()
        for r in results:
            writer.writerow({**asdict(r), "status": r.status.value})


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch manifest PDFs (Unpaywall, then landing page).")
    parser.add_argument("--email", required=True, help="contact address Unpaywall requires; never stored")
    parser.add_argument("--manifest", default=_DEFAULT_MANIFEST)
    parser.add_argument("--pdf-dir", default=_DEFAULT_PDF_DIR)
    parser.add_argument("--report", default=_DEFAULT_REPORT)
    args = parser.parse_args(argv)

    results = fetch_all(load_manifest(args.manifest), args.pdf_dir, args.email)
    write_report(results, args.report)

    for status in FetchStatus:
        print(f"{status.value:12s} {sum(r.status == status for r in results)}")
    return 0


# ── per entry ───────────────────────────────────────────────────

def _fetch_one(entry: ManifestEntry, pdf_dir: str, email: str) -> FetchResult:
    path = os.path.join(pdf_dir, entry.pdf_filename)

    # Idempotent: a valid file on disk is never fetched again
    existing = _read_valid_pdf(path)
    if existing is not None:
        return FetchResult(entry.doi, FetchStatus.PRESENT, _Source.DISK.value, _sha256(existing))

    # Sources in order; the landing page is only requested when Unpaywall gave nothing usable
    finders = [
        (_Source.UNPAYWALL, lambda: _unpaywall_pdf_url(entry.doi, email)),
        (_Source.LANDING_PAGE, lambda: _landing_page_pdf_url(entry.doi)),
    ]
    refusals = []
    for source, find_url in finders:
        url = find_url()
        if not url:
            continue

        status, content_type, body = _get(url)
        if status == _HTTP_OK and body.startswith(_PDF_MAGIC):
            _write_atomic(path, body)
            return FetchResult(entry.doi, FetchStatus.DOWNLOADED, source.value, _sha256(body), url)
        refusals.append((source, url, status, content_type))

    return _failure(entry.doi, refusals)


def _failure(doi: str, refusals: list[tuple]) -> FetchResult:
    """PDF URL found but refused → PAYWALLED; claimed a PDF but wasn't → INVALID_PDF; else NOT_FOUND."""
    if not refusals:
        return FetchResult(doi, FetchStatus.NOT_FOUND, detail="no PDF URL from Unpaywall or landing page")

    source, url, status, content_type = refusals[-1]
    detail = f"HTTP {status} {content_type} {url}"
    if status == _HTTP_OK and content_type.startswith(_PDF_CONTENT_TYPE):
        return FetchResult(doi, FetchStatus.INVALID_PDF, source.value, detail=detail)
    if status in _HTTP_REFUSED or status == _HTTP_OK:
        return FetchResult(doi, FetchStatus.PAYWALLED, source.value, detail=detail)
    return FetchResult(doi, FetchStatus.NOT_FOUND, source.value, detail=detail)


def _unpaywall_pdf_url(doi: str, email: str) -> Optional[str]:
    # The only request that carries the email
    url = _UNPAYWALL_URL.format(doi=urllib.parse.quote(doi, safe="/"), email=urllib.parse.quote(email))
    status, _, body = _get(url)
    if status != _HTTP_OK:
        return None
    try:
        location = json.loads(body.decode("utf-8")).get("best_oa_location") or {}
    except ValueError:
        return None
    return location.get("url_for_pdf")


def _landing_page_pdf_url(doi: str) -> Optional[str]:
    """<meta name="citation_pdf_url" content="…"> on the publisher page (works inside a VPN)."""
    landing = _DOI_RESOLVER.format(doi=urllib.parse.quote(doi, safe="/"))
    status, _, body = _get(landing)
    if status != _HTTP_OK:
        return None
    finder = _CitationPdfFinder()
    finder.feed(body.decode("utf-8", errors="replace"))
    return urllib.parse.urljoin(landing, finder.url) if finder.url else None


class _CitationPdfFinder(HTMLParser):
    def __init__(self):
        super().__init__()
        self.url: Optional[str] = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if self.url is None and tag == "meta" and attributes.get("name") == "citation_pdf_url":
            self.url = attributes.get("content")


# ── I/O ─────────────────────────────────────────────────────────

def _get(url: str) -> tuple[int, str, bytes]:
    # Polite pacing between consecutive requests
    response = _http_get(url)
    time.sleep(_REQUEST_DELAY_S)
    return response


def _http_get(url: str) -> tuple[int, str, bytes]:
    """The single HTTP seam: (status, content type, body); status 0 on a network error."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            return response.status, response.headers.get_content_type(), response.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get_content_type() if e.headers else "", b""
    except (urllib.error.URLError, OSError) as e:
        return 0, "", str(e).encode()


def _read_valid_pdf(path: str) -> Optional[bytes]:
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        body = f.read()
    return body if body.startswith(_PDF_MAGIC) else None


def _write_atomic(path: str, body: bytes) -> None:
    """Write <name>.part, then rename: a crash never leaves a half PDF under the real name."""
    part = path + _PART_SUFFIX
    try:
        with open(part, "wb") as f:
            f.write(body)
        os.replace(part, path)
    finally:
        if os.path.exists(part):
            os.remove(part)


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


if __name__ == "__main__":
    sys.exit(main())
