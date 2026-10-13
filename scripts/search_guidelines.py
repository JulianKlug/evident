"""Search PubMed for ESICM, SCCM and ACP guidelines; merge hits into the candidates file.

    three queries per society ──esearch──▶ PMIDs ──esummary──▶ (doi, year, title, journal)
      title:   society journal + "guideline" in the title
      broad:   society journal + recommendation/consensus/statement titles or guideline publication type
      society: society name in title or corporate author, any journal (co-led guidelines elsewhere)
        │
        ▼
    corpus/candidates.csv: existing rows keep their decision; new hits are appended undecided,
    tagged with the query that found them first

A human then sets decision (include/exclude) and reason per row (spec M3 §4.1–4.3).

Usage:
    python scripts/search_guidelines.py [--email YOU@ORG] [--until 2026-10-04] [--out corpus/candidates.csv]

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
from datetime import date, datetime
from enum import Enum
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
_DATE_FORMAT = "%Y-%m-%d"


class Source(str, Enum):
    """Which query found a candidate first; stored in the source column."""
    TITLE = "pubmed"            # the M3 query: "guideline" in the title
    BROAD = "pubmed_broad"      # e.g. "ESICM consensus and clinical practice recommendations" (2018)
    SOCIETY = "pubmed_society"  # e.g. ATS/ESICM/SCCM ARDS guideline in Am J Respir Crit Care Med (2017)


@dataclass(frozen=True)
class Query:
    society: str
    source: Source
    term: str


_JOURNALS = {
    "ESICM": '"Intensive Care Med"[ta]',
    "SCCM": '("Crit Care Med"[ta] OR "Pediatr Crit Care Med"[ta])',
    "ACP": '"Ann Intern Med"[ta]',
}
_NAMES = {
    "ESICM": '("European Society of Intensive Care Medicine"[ti] OR ESICM[ti] '
             'OR "European Society of Intensive Care Medicine"[cn])',
    "SCCM": '("Society of Critical Care Medicine"[ti] OR SCCM[ti] OR "American College of Critical Care Medicine"[ti] '
            'OR "Surviving Sepsis Campaign"[ti] OR "Society of Critical Care Medicine"[cn])',
    "ACP": '("American College of Physicians"[ti] OR "American College of Physicians"[cn])',
}
_RECOMMENDATION_TITLES = ('(recommendation*[ti] OR consensus[ti] OR statement*[ti] OR "practice parameter*"[ti] '
                          'OR guideline[pt] OR "practice guideline"[pt])')
_GUIDANCE_TITLES = f'(guideline*[ti] OR {_RECOMMENDATION_TITLES[1:]}'

# Quoted verbatim in the paper's methods; the date filter is appended per run
QUERIES = (
    [Query("ESICM", Source.TITLE, f'{_JOURNALS["ESICM"]} AND guideline*[ti]'),
     Query("SCCM", Source.TITLE, f'{_JOURNALS["SCCM"]} AND guideline*[ti]'),
     Query("ACP", Source.TITLE, f'{_JOURNALS["ACP"]} AND "American College of Physicians"[ti]')]
    + [Query(s, Source.BROAD, f"{j} AND {_RECOMMENDATION_TITLES}") for s, j in _JOURNALS.items()]
    + [Query(s, Source.SOCIETY, f"{n} AND {_GUIDANCE_TITLES}") for s, n in _NAMES.items()]
)

COLUMNS = ("doi", "society", "year", "title", "journal", "pmid", "source", "decision", "reason", "duplicate_of")


@dataclass(frozen=True)
class Hit:
    doi: str
    society: str
    year: Optional[int]
    title: str
    journal: str
    pmid: str
    source: str


def query_for(query: Query, until: date) -> str:
    """Query term + publication date window, e.g. '… AND ("2016/01/01"[dp] : "2026/10/04"[dp])'."""
    return f'{query.term} AND ("{_START_DATE}"[dp] : "{until:%Y/%m/%d}"[dp])'


def search(query: Query, email: Optional[str], until: date) -> tuple[list[Hit], list[str]]:
    """Hits with a DOI, and PMIDs without one (to resolve by hand)."""
    pmids = _esearch(query_for(query, until), email)
    hits, no_doi = [], []
    for start in range(0, len(pmids), _SUMMARY_BATCH):
        for record in _esummary(pmids[start:start + _SUMMARY_BATCH], email):
            hit = parse_summary(record, query)
            if hit is None:
                no_doi.append(str(record.get("uid", "")))
                continue
            hits.append(hit)
    return hits, no_doi


def parse_summary(record: dict, query: Query) -> Optional[Hit]:
    """esummary record → Hit; year from the electronic date, else the print date; None without a DOI."""
    doi = next((a.get("value", "") for a in record.get("articleids", []) if a.get("idtype") == "doi"), "")
    if not doi:
        return None
    return Hit(
        doi=normalize_doi(doi),
        society=query.society,
        year=_year(record.get("epubdate", "")) or _year(record.get("pubdate", "")),
        title=record.get("title", "").strip(),
        journal=record.get("source", ""),
        pmid=str(record.get("uid", "")),
        source=query.source.value,
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
                     "title": h.title, "journal": h.journal, "pmid": h.pmid, "source": h.source,
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


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PubMed search for MVP guidelines → candidates file.")
    parser.add_argument("--email", help="contact address NCBI recommends (optional); never stored")
    parser.add_argument("--until", type=lambda v: datetime.strptime(v, _DATE_FORMAT).date(), default=date.today(),
                        help="last publication date searched (default: today); reproduces a past search")
    parser.add_argument("--out", default=_DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # Title queries first, so a hit found by several queries keeps the oldest source
    all_hits = []
    for query in QUERIES:
        hits, no_doi = search(query, args.email, args.until)
        all_hits.extend(hits)
        print(f"{query.society} {query.source.value}: {len(hits)} hits with a DOI; "
              f"without DOI (resolve by hand): {', '.join(no_doi) or 'none'}")
        print(f"  query: {query_for(query, args.until)}")

    existing = read_rows(args.out)
    rows = merge(existing, all_hits)
    write_rows(rows, args.out)
    print(f"{args.out}: {len(rows)} rows ({len(rows) - len(existing)} new), searched until {args.until:%Y-%m-%d}")
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
