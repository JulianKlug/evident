"""Corpus manifest: load + validate, resolve current editions, sync to the store.

    corpus/manifest.csv ──load_manifest──▶ [ManifestEntry] ──sync_to_store──▶ Store
                                               │
                                       current_editions → {doi}
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from evident.domain import GradingFamily, Guideline
from evident.store import Store

_KNOWN_SOCIETIES = {"ESICM", "SCCM", "ACP"}
_MIN_YEAR = 1990
_DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$")
_TOPIC_PATTERN = re.compile(r"^[a-z0-9-]+$")
_DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "doi:")
_HASH_CHUNK_BYTES = 1 << 20
_FIRST_DATA_LINE = 2  # line 1 is the header

_REQUIRED_COLUMNS = ("doi", "society", "year", "title", "topic_id", "grading_family")
_OPTIONAL_COLUMNS = ("supersedes_doi", "pdf_filename")


class ManifestError(ValueError):
    """All problems found in a manifest, each prefixed with its line number."""

    def __init__(self, problems: list[str]):
        super().__init__("Invalid manifest:\n" + "\n".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class ManifestEntry:
    doi: str
    society: str
    year: int
    title: str
    topic_id: str
    grading_family: GradingFamily
    pdf_filename: str
    supersedes_doi: Optional[str] = None
    line: int = 0  # source line, for error messages


@dataclass
class SyncReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    missing_pdf: list[str] = field(default_factory=list)


def load_manifest(path: str) -> list[ManifestEntry]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in _REQUIRED_COLUMNS + _OPTIONAL_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ManifestError([f"line 1: missing column(s) {', '.join(missing)}"])
        rows = list(reader)

    problems: list[str] = []
    entries = []
    for line, row in enumerate(rows, start=_FIRST_DATA_LINE):
        entry = _parse_row(row, line, problems)
        if entry:
            entries.append(entry)

    problems.extend(_cross_row_problems(entries))
    if problems:
        raise ManifestError(problems)
    return entries


def current_editions(entries: list[ManifestEntry]) -> set[str]:
    superseded = {e.supersedes_doi for e in entries if e.supersedes_doi}

    # Latest non-superseded edition per topic; a tie is ambiguous
    by_topic: dict[str, list[ManifestEntry]] = {}
    for e in entries:
        if e.doi not in superseded:
            by_topic.setdefault(e.topic_id, []).append(e)

    current = set()
    problems = []
    for topic, candidates in by_topic.items():
        latest_year = max(e.year for e in candidates)
        latest = [e for e in candidates if e.year == latest_year]
        if len(latest) > 1:
            lines = ", ".join(str(e.line) for e in latest)
            problems.append(f"lines {lines}: ambiguous current edition for topic '{topic}' ({latest_year})")
            continue
        current.add(latest[0].doi)

    if problems:
        raise ManifestError(problems)
    return current


def sync_to_store(entries: list[ManifestEntry], store: Store, pdf_dir: str) -> SyncReport:
    report = SyncReport()
    for e in entries:
        pdf_path = os.path.join(pdf_dir, e.pdf_filename)
        pdf_present = os.path.isfile(pdf_path)
        if not pdf_present:
            report.missing_pdf.append(e.doi)

        existed = store.get_guideline(e.doi) is not None
        store.upsert_guideline(Guideline(
            doi=e.doi, society=e.society, year=e.year, title=e.title, topic_id=e.topic_id,
            grading_family=e.grading_family, supersedes_doi=e.supersedes_doi,
            pdf_path=pdf_path if pdf_present else None,
            pdf_sha256=_sha256(pdf_path) if pdf_present else None,
        ))
        (report.updated if existed else report.added).append(e.doi)
    return report


def _parse_row(row: dict, line: int, problems: list[str]) -> Optional[ManifestEntry]:
    """Validate one row; append every problem found; None if any."""
    n_before = len(problems)

    def problem(msg: str) -> None:
        problems.append(f"line {line}: {msg}")

    def value(column: str) -> str:
        return (row.get(column) or "").strip()

    raw_doi = _strip_doi_prefix(value("doi"))
    doi = raw_doi.lower()
    if not _DOI_PATTERN.match(doi):
        problem(f"invalid doi '{value('doi')}'")

    society = value("society")
    if society not in _KNOWN_SOCIETIES:
        problem(f"unknown society '{society}'")

    year = _parse_year(value("year"))
    if year is None:
        problem(f"year '{value('year')}' not an integer in {_MIN_YEAR}..{date.today().year}")

    title = value("title")
    if not title:
        problem("empty title")

    topic_id = value("topic_id")
    if not _TOPIC_PATTERN.match(topic_id):
        problem(f"topic_id '{topic_id}' is not a slug")

    family = _parse_family(value("grading_family"))
    if family is None:
        problem(f"unknown grading_family '{value('grading_family')}'")

    if len(problems) > n_before:
        return None

    supersedes = _strip_doi_prefix(value("supersedes_doi")).lower() or None
    return ManifestEntry(
        doi=doi, society=society, year=year, title=title, topic_id=topic_id,
        grading_family=family, supersedes_doi=supersedes, line=line,
        # Filename keeps the DOI's original case, matching existing PDF names
        pdf_filename=value("pdf_filename") or _doi_to_filename(raw_doi),
    )


def _cross_row_problems(entries: list[ManifestEntry]) -> list[str]:
    problems = []
    by_doi: dict[str, ManifestEntry] = {}
    for e in entries:
        if e.doi in by_doi:
            problems.append(f"line {e.line}: duplicate doi {e.doi} (first on line {by_doi[e.doi].line})")
            continue
        by_doi[e.doi] = e

    # supersedes_doi: in manifest, same topic, earlier year
    for e in entries:
        if not e.supersedes_doi:
            continue
        old = by_doi.get(e.supersedes_doi)
        if old is None:
            problems.append(f"line {e.line}: supersedes unknown doi {e.supersedes_doi}")
            continue
        if old.topic_id != e.topic_id:
            problems.append(f"line {e.line}: supersedes {old.doi} from another topic '{old.topic_id}'")
        if old.year >= e.year:
            problems.append(f"line {e.line}: supersedes {old.doi} which is not earlier ({old.year})")

    problems.extend(_cycle_problems(entries, by_doi))
    return problems


def _cycle_problems(entries: list[ManifestEntry], by_doi: dict[str, ManifestEntry]) -> list[str]:
    """Walk each supersession chain; revisiting a DOI means a cycle."""
    problems = []
    reported: set[str] = set()
    for start in entries:
        seen = []
        current: Optional[ManifestEntry] = start
        while current and current.supersedes_doi:
            if current.doi in seen:
                cycle = frozenset(seen)
                if not cycle & reported:
                    problems.append(f"line {start.line}: supersession cycle {' -> '.join(seen)}")
                    reported |= cycle
                break
            seen.append(current.doi)
            current = by_doi.get(current.supersedes_doi)
    return problems


def _parse_year(raw: str) -> Optional[int]:
    try:
        year = int(raw)
    except ValueError:
        return None
    return year if _MIN_YEAR <= year <= date.today().year else None


def _parse_family(raw: str) -> Optional[GradingFamily]:
    try:
        return GradingFamily(raw.lower())
    except ValueError:
        return None


def _strip_doi_prefix(doi: str) -> str:
    """Same prefixes as the store's DOI normalization; case is kept for filenames."""
    for prefix in _DOI_PREFIXES:
        if doi.lower().startswith(prefix):
            return doi[len(prefix):]
    return doi


def _doi_to_filename(doi: str) -> str:
    # Same rule as extraction/pdf_loader.py; evident must not import extraction
    return doi.replace("/", "_").replace(".", "_") + ".pdf"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()
