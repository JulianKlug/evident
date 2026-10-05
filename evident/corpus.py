"""Corpus manifest: load + validate, resolve current editions, sync to the store.

    corpus/manifest.csv ──load_manifest──▶ [ManifestEntry] ──sync_to_store──▶ Store
                                               │
                                       current_editions → {doi}

    corpus/candidates.csv ──load_candidates──▶ [Candidate] ──check_candidates(manifest)
    (every search hit, with an include/exclude decision; curation record for the supplement)
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional

from evident.domain import GradingFamily, Guideline
from evident.store import Store
from utils.doi import doi_to_filename, normalize_doi, strip_doi_prefix

_KNOWN_SOCIETIES = {"ESICM", "SCCM", "ACP"}
_MIN_YEAR = 1990
_DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$")
_TOPIC_PATTERN = re.compile(r"^[a-z0-9-]+$")
_HASH_CHUNK_BYTES = 1 << 20
_FIRST_DATA_LINE = 2  # line 1 is the header

_REQUIRED_COLUMNS = ("doi", "society", "year", "title", "topic_id", "grading_family")
_OPTIONAL_COLUMNS = ("supersedes_doi", "pdf_filename")
_JOINT_COLUMN = "joint_with"  # may be absent from the header
_JOINT_SEPARATOR = ";"

_CANDIDATE_COLUMNS = ("doi", "society", "year", "title", "journal", "pmid", "source", "decision", "reason",
                      "duplicate_of")


class CandidateDecision(str, Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"


class ExclusionReason(str, Enum):
    NOT_GUIDELINE = "not_guideline"                  # review, research agenda, methods paper
    NOT_GRADED = "not_graded"                        # consensus, Delphi, best practice advice
    OTHER_SOCIETY = "other_society"                  # led by a body outside the MVP societies
    ENDORSEMENT = "endorsement"                      # endorses another body's guideline
    DERIVATIVE = "derivative"                        # patient summary, executive summary
    DUPLICATE_PUBLICATION = "duplicate_publication"  # second journal copy; duplicate_of names the kept one
    OUT_OF_RANGE = "out_of_range"                    # first online before 2016
    PDF_UNAVAILABLE = "pdf_unavailable"              # no PDF after fetch and a manual attempt


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
    joint_with: tuple[str, ...] = ()  # other MVP societies co-leading it, e.g. ("ESICM",)
    line: int = 0  # source line, for error messages


@dataclass(frozen=True)
class Candidate:
    doi: str
    society: str
    year: Optional[int]
    title: str
    decision: CandidateDecision
    reason: Optional[ExclusionReason]
    duplicate_of: Optional[str]
    line: int = 0


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


def load_candidates(path: str) -> list[Candidate]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in _CANDIDATE_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ManifestError([f"line 1: missing column(s) {', '.join(missing)}"])
        rows = list(reader)

    problems: list[str] = []
    candidates = []
    for line, row in enumerate(rows, start=_FIRST_DATA_LINE):
        candidate = _parse_candidate(row, line, problems)
        if candidate:
            candidates.append(candidate)

    # A duplicate publication points at a kept (included) copy
    included = {c.doi for c in candidates if c.decision == CandidateDecision.INCLUDE}
    for c in candidates:
        if c.reason == ExclusionReason.DUPLICATE_PUBLICATION and c.duplicate_of not in included:
            problems.append(f"line {c.line}: duplicate_of '{c.duplicate_of or ''}' is not an included doi")

    if problems:
        raise ManifestError(problems)
    return candidates


def check_candidates(candidates: list[Candidate], entries: list[ManifestEntry]) -> None:
    """Included candidates and manifest rows must be the same set of DOIs."""
    included = {c.doi for c in candidates if c.decision == CandidateDecision.INCLUDE}
    in_manifest = {e.doi for e in entries}
    problems = [f"included candidate {d} has no manifest row" for d in sorted(included - in_manifest)]
    problems += [f"manifest row {d} is not an included candidate" for d in sorted(in_manifest - included)]
    if problems:
        raise ManifestError(problems)


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

    raw_doi = strip_doi_prefix(value("doi"))
    doi = normalize_doi(raw_doi)
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

    joint_with = tuple(j.strip() for j in value(_JOINT_COLUMN).split(_JOINT_SEPARATOR) if j.strip())
    for joint in joint_with:
        if joint not in _KNOWN_SOCIETIES or joint == society:
            problem(f"joint_with '{joint}' is not another known society")

    if len(problems) > n_before:
        return None

    supersedes = normalize_doi(value("supersedes_doi")) or None
    return ManifestEntry(
        doi=doi, society=society, year=year, title=title, topic_id=topic_id,
        grading_family=family, supersedes_doi=supersedes, joint_with=joint_with, line=line,
        # Filename keeps the DOI's original case, matching existing PDF names
        pdf_filename=value("pdf_filename") or doi_to_filename(raw_doi),
    )


def _parse_candidate(row: dict, line: int, problems: list[str]) -> Optional[Candidate]:
    """Validate one candidate row; append every problem found; None if any."""
    n_before = len(problems)

    def problem(msg: str) -> None:
        problems.append(f"line {line}: {msg}")

    def value(column: str) -> str:
        return (row.get(column) or "").strip()

    doi = normalize_doi(strip_doi_prefix(value("doi")))
    if not _DOI_PATTERN.match(doi):
        problem(f"invalid doi '{value('doi')}'")

    decision = _parse_enum(CandidateDecision, value("decision"))
    if decision is None:
        problem(f"decision '{value('decision')}' is not include/exclude")

    reason = _parse_enum(ExclusionReason, value("reason"))
    if decision == CandidateDecision.EXCLUDE and reason is None:
        problem(f"exclude needs a reason, got '{value('reason')}'")
    if decision == CandidateDecision.INCLUDE and value("reason"):
        problem("include must not have a reason")

    if len(problems) > n_before:
        return None

    duplicate_of = normalize_doi(strip_doi_prefix(value("duplicate_of"))) or None
    return Candidate(doi=doi, society=value("society"), year=_parse_year(value("year")), title=value("title"),
                     decision=decision, reason=reason, duplicate_of=duplicate_of, line=line)


def _parse_enum(enum, raw: str):
    try:
        return enum(raw.lower())
    except ValueError:
        return None


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


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()
