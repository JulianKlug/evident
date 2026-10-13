"""Fetch the PDF of every manifest guideline into the PDF dir; write a report of what is missing.

    manifest entry ──▶ already on disk? ──yes──▶ PRESENT
                           │ no
                           ▼
                      legal open copies, each URL tried once, in order:   ──PDF──▶ DOWNLOADED
                        Unpaywall oa_locations (only with --email)
                        Europe PMC open-access PMC copy
                        Semantic Scholar openAccessPdf
                           │ none / refused
                           ▼
                      https://doi.org/<doi> → <meta citation_pdf_url> ──PDF──▶ DOWNLOADED
                           │ none / refused
                           ▼
                      PAYWALLED | INVALID_PDF | NOT_FOUND
                      BLOCKED: the landing page refused a script (e.g. Cloudflare 403); a browser may pass

Usage (inside the institutional VPN for paywalled journals):
    python scripts/fetch_pdfs.py [--email YOU@ORG] [--manifest corpus/manifest.csv] [--pdf-dir DIR]

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
_EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?{query}"
_EUROPE_PMC_PDF = "https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
_SEMANTIC_SCHOLAR = "https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=openAccessPdf"
_OPEN_ACCESS_FLAG = "Y"
_PDF_MAGIC = b"%PDF"
_PDF_CONTENT_TYPE = "application/pdf"
_HTTP_OK = 200
_HTTP_REFUSED = {401, 403}
_HTTP_BLOCKED = {403, 429, 503}  # bot protection on the landing page, e.g. Cloudflare "Just a moment..."
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
    BLOCKED = "blocked"


class _Source(str, Enum):
    DISK = "disk"
    UNPAYWALL = "unpaywall"
    EUROPE_PMC = "europe_pmc"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    LANDING_PAGE = "landing_page"


@dataclass(frozen=True)
class FetchResult:
    doi: str
    status: FetchStatus
    source: str = ""
    sha256: str = ""
    detail: str = ""


def fetch_all(entries: list[ManifestEntry], pdf_dir: str, email: Optional[str]) -> list[FetchResult]:
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
    parser.add_argument("--email", help="contact address Unpaywall requires; without it Unpaywall is skipped; "
                                        "never stored")
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

def _fetch_one(entry: ManifestEntry, pdf_dir: str, email: Optional[str]) -> FetchResult:
    path = os.path.join(pdf_dir, entry.pdf_filename)

    # Idempotent: a valid file on disk is never fetched again
    existing = _read_valid_pdf(path)
    if existing is not None:
        return FetchResult(entry.doi, FetchStatus.PRESENT, _Source.DISK.value, _sha256(existing))

    # Sources in order; each is only asked when the ones before gave nothing usable
    landing_status = []  # HTTP status of the landing page, once requested

    def landing_page_urls() -> list[str]:
        url, status = _landing_page_pdf_url(entry.doi)
        landing_status.append(status)
        return [url] if url else []

    finders = [
        (_Source.UNPAYWALL, lambda: _unpaywall_pdf_urls(entry.doi, email) if email else []),
        (_Source.EUROPE_PMC, lambda: _europe_pmc_pdf_urls(entry.doi)),
        (_Source.SEMANTIC_SCHOLAR, lambda: _semantic_scholar_pdf_urls(entry.doi)),
        (_Source.LANDING_PAGE, landing_page_urls),
    ]
    refusals = []
    tried: set[str] = set()  # one request per URL, even when several sources list it
    for source, find_urls in finders:
        for url in find_urls():
            if url in tried:
                continue
            tried.add(url)

            status, content_type, body = _get(url)
            if status == _HTTP_OK and body.startswith(_PDF_MAGIC):
                _write_atomic(path, body)
                return FetchResult(entry.doi, FetchStatus.DOWNLOADED, source.value, _sha256(body), url)
            refusals.append((source, url, status, content_type))

    return _failure(entry.doi, refusals, landing_status[0] if landing_status else None)


def _failure(doi: str, refusals: list[tuple], landing_status: Optional[int]) -> FetchResult:
    """PDF URL found but refused → PAYWALLED; claimed a PDF but wasn't → INVALID_PDF;
    no PDF URL because the landing page refused → BLOCKED; else NOT_FOUND."""
    if not refusals:
        landing = _DOI_RESOLVER.format(doi=doi)
        if landing_status in _HTTP_BLOCKED:
            return FetchResult(doi, FetchStatus.BLOCKED, _Source.LANDING_PAGE.value,
                               detail=f"landing page HTTP {landing_status} {landing}")
        if landing_status == _HTTP_OK:
            return FetchResult(doi, FetchStatus.NOT_FOUND, _Source.LANDING_PAGE.value,
                               detail=f"no open-access PDF; landing page has no citation_pdf_url {landing}")
        return FetchResult(doi, FetchStatus.NOT_FOUND, detail="no PDF URL from any source")

    source, url, status, content_type = refusals[-1]
    detail = f"HTTP {status} {content_type} {url}"
    if status == _HTTP_OK and content_type.startswith(_PDF_CONTENT_TYPE):
        return FetchResult(doi, FetchStatus.INVALID_PDF, source.value, detail=detail)
    if status in _HTTP_REFUSED or status == _HTTP_OK:
        return FetchResult(doi, FetchStatus.PAYWALLED, source.value, detail=detail)
    return FetchResult(doi, FetchStatus.NOT_FOUND, source.value, detail=detail)


def _unpaywall_pdf_urls(doi: str, email: str) -> list[str]:
    """url_for_pdf of every open-access location, best first."""
    # The only request that carries the email
    url = _UNPAYWALL_URL.format(doi=urllib.parse.quote(doi, safe="/"), email=urllib.parse.quote(email))
    record = _get_json(url)
    best = record.get("best_oa_location") or {}
    locations = [best] + (record.get("oa_locations") or [])
    return _unique([loc.get("url_for_pdf") for loc in locations if loc])


def _europe_pmc_pdf_urls(doi: str) -> list[str]:
    """The PMC copy rendered by Europe PMC, only for open-access records (author manuscripts are refused)."""
    query = urllib.parse.urlencode({"query": f'DOI:"{doi}"', "format": "json", "resultType": "core"})
    results = (_get_json(_EUROPE_PMC_SEARCH.format(query=query)).get("resultList") or {}).get("result") or []
    return _unique([_EUROPE_PMC_PDF.format(pmcid=r["pmcid"]) for r in results
                    if r.get("pmcid") and r.get("isOpenAccess") == _OPEN_ACCESS_FLAG])


def _semantic_scholar_pdf_urls(doi: str) -> list[str]:
    record = _get_json(_SEMANTIC_SCHOLAR.format(doi=urllib.parse.quote(doi, safe="/")))
    return _unique([(record.get("openAccessPdf") or {}).get("url")])


def _get_json(url: str) -> dict:
    """Parsed JSON body; {} on any HTTP or parse failure (a source with nothing to offer)."""
    status, _, body = _get(url)
    if status != _HTTP_OK:
        return {}
    try:
        parsed = json.loads(body.decode("utf-8"))
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _unique(urls: list[Optional[str]]) -> list[str]:
    """Drop empties and repeats, keep order."""
    seen: list[str] = []
    for u in urls:
        if u and u not in seen:
            seen.append(u)
    return seen


def _landing_page_pdf_url(doi: str) -> tuple[Optional[str], int]:
    """(<meta name="citation_pdf_url"> on the publisher page, landing page HTTP status)."""
    landing = _DOI_RESOLVER.format(doi=urllib.parse.quote(doi, safe="/"))
    status, _, body = _get(landing)
    if status != _HTTP_OK:
        return None, status
    finder = _CitationPdfFinder()
    finder.feed(body.decode("utf-8", errors="replace"))
    return (urllib.parse.urljoin(landing, finder.url) if finder.url else None), status


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
