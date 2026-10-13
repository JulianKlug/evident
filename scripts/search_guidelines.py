"""Search PubMed for ESICM, SCCM and ACP guidelines; merge hits into the candidates file.

    one query per society ──esearch──▶ PMIDs ──esummary──▶ (doi, year, title, journal)
        │
        ▼
    corpus/candidates.csv: existing rows keep their decision; new hits are appended undecided

A human then sets decision (include/exclude) and reason per row (spec M3 §4.1–4.3).

Usage:
    python scripts/search_guidelines.py [--email YOU@ORG] [--out corpus/candidates.csv]

Never imports extraction.pdf_loader (its DOI fallback reaches Sci-Hub).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.doi import normalize_doi  # noqa: E402

_DEFAULT_OUT = "corpus/candidates.csv"
_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
_TOOL = "evident"
_START_DATE = "2016/01/01"
_RETMAX = 2000
_SUMMARY_BATCH = 200
_REQUEST_DELAY_S = 0.4  # NCBI: at most 3 requests/s without an API key
_TIMEOUT_S = 60
_HTTP_OK = 200
_SOURCE_PUBMED = "pubmed"

# Quoted verbatim in the paper's methods; the date filter is appended per run
QUERIES = {
    "ESICM": '"Intensive Care Med"[ta] AND guideline*[ti]',
    "SCCM": '("Crit Care Med"[ta] OR "Pediatr Crit Care Med"[ta]) AND guideline*[ti]',
    "ACP": '"Ann Intern Med"[ta] AND "American College of Physicians"[ti]',
}

COLUMNS = ("doi", "society", "year", "title", "journal", "pmid", "source", "decision", "reason", "duplicate_of")


@dataclass(frozen=True)
class Hit:
    doi: str
    society: str
    year: Optional[int]
    title: str
    journal: str
    pmid: str


def query_for(society: str, today: date) -> str:
    """Society query + publication date window, e.g. '… AND ("2016/01/01"[dp] : "2026/10/04"[dp])'."""
    return f'{QUERIES[society]} AND ("{_START_DATE}"[dp] : "{today:%Y/%m/%d}"[dp])'


def search(society: str, email: Optional[str], today: date) -> tuple[list[Hit], list[str]]:
    """Hits with a DOI, and PMIDs without one (to resolve by hand)."""
    pmids = _esearch(query_for(society, today), email)
    hits, no_doi = [], []
    for start in range(0, len(pmids), _SUMMARY_BATCH):
        for record in _esummary(pmids[start:start + _SUMMARY_BATCH], email):
            hit = parse_summary(record, society)
            if hit is None:
                no_doi.append(str(record.get("uid", "")))
                continue
            hits.append(hit)
    return hits, no_doi


def parse_summary(record: dict, society: str) -> Optional[Hit]:
    """esummary record → Hit; year from the electronic date, else the print date; None without a DOI."""
    doi = next((a.get("value", "") for a in record.get("articleids", []) if a.get("idtype") == "doi"), "")
    if not doi:
        return None
    return Hit(
        doi=normalize_doi(doi),
        society=society,
        year=_year(record.get("epubdate", "")) or _year(record.get("pubdate", "")),
        title=record.get("title", "").strip(),
        journal=record.get("source", ""),
        pmid=str(record.get("uid", "")),
    )


def merge(existing: list[dict], hits: list[Hit]) -> list[dict]:
    """Existing rows unchanged (decisions kept); hits with a new DOI appended undecided; sorted."""
    known = {normalize_doi(r["doi"]) for r in existing}
    rows = [dict(r) for r in existing]
    for h in hits:
        if h.doi in known:
            continue
        known.add(h.doi)
        rows.append({"doi": h.doi, "society": h.society, "year": "" if h.year is None else str(h.year),
                     "title": h.title, "journal": h.journal, "pmid": h.pmid, "source": _SOURCE_PUBMED,
                     "decision": "", "reason": "", "duplicate_of": ""})
    return sorted(rows, key=lambda r: (r["society"], r["year"], r["doi"]))


def read_rows(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{c: row.get(c) or "" for c in COLUMNS} for row in csv.DictReader(f)]


def write_rows(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PubMed search for MVP guidelines → candidates file.")
    parser.add_argument("--email", help="contact address NCBI recommends (optional); never stored")
    parser.add_argument("--out", default=_DEFAULT_OUT)
    args = parser.parse_args(argv)

    today = date.today()
    all_hits = []
    for society in QUERIES:
        hits, no_doi = search(society, args.email, today)
        all_hits.extend(hits)
        print(f"{society}: {len(hits)} hits with a DOI; without DOI (resolve by hand): {', '.join(no_doi) or 'none'}")
        print(f"  query: {query_for(society, today)}")

    existing = read_rows(args.out)
    rows = merge(existing, all_hits)
    write_rows(rows, args.out)
    print(f"{args.out}: {len(rows)} rows ({len(rows) - len(existing)} new), search date {today:%Y-%m-%d}")
    return 0


# ── E-utilities ─────────────────────────────────────────────────

def _esearch(term: str, email: Optional[str]) -> list[str]:
    body = _get_json("esearch.fcgi", {"db": "pubmed", "term": term, "retmax": _RETMAX, "retmode": "json"}, email)
    return body.get("esearchresult", {}).get("idlist", [])


def _esummary(pmids: list[str], email: Optional[str]) -> list[dict]:
    if not pmids:
        return []
    body = _get_json("esummary.fcgi", {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"}, email)
    result = body.get("result", {})
    return [result[uid] for uid in result.get("uids", []) if uid in result]


def _get_json(endpoint: str, params: dict, email: Optional[str]) -> dict:
    contact = {"email": email} if email else {}
    query = urllib.parse.urlencode({**params, "tool": _TOOL, **contact})
    status, body = _http_get(f"{_EUTILS}{endpoint}?{query}")
    time.sleep(_REQUEST_DELAY_S)
    if status != _HTTP_OK:
        raise RuntimeError(f"PubMed {endpoint} returned HTTP {status}")
    return json.loads(body.decode("utf-8"))


def _http_get(url: str) -> tuple[int, bytes]:
    """The single HTTP seam: (status, body); status 0 on a network error."""
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except (urllib.error.URLError, OSError) as e:
        return 0, str(e).encode()


def _year(raw: str) -> Optional[int]:
    """'2021 Nov' → 2021; '' → None."""
    head = raw.strip()[:4]
    return int(head) if head.isdigit() else None


if __name__ == "__main__":
    sys.exit(main())
