"""Within-guideline duplicate recommendations: screen, frozen LLM verdicts, clusters (spec §5 amendment).

    rows of one guideline ─┬─ BioLORD cosine ≥ 0.75 ─┐
                           └─ lexical ratio ≥ 0.70 ──┴─▶ screen ─▶ [CandidatePair]
                                                                     │ scripts/paper1_dedup_adjudicate.py (LLM, offline)
                                                                     ▼
    corpus/dedup/paper1_verdicts.csv ─▶ load_adjudications ─▶ check_complete (every screened pair, nothing else)
                                                          └─▶ deduplicate ─▶ kept rows (one per cluster)

Pure apart from the CSV read/write. publish never calls an LLM: it recomputes the screen and
fails closed when a screened pair lacks a verdict (DedupVerdictError).
"""

from __future__ import annotations

import csv
import difflib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from evident.analytics import HarmonizedRow
from evident.domain import AxisStatus, Category

SCREEN_BIOLORD = 0.75  # cosine; abbreviation copies ("NMBA" vs "neuromuscular-blocking agent") score 0.6–0.8
SCREEN_LEXICAL = 0.70  # character ratio of the normalized texts

# "(strong recommendation; high-certainty evidence)." at the end of a recommendation
_GRADE_TAIL = re.compile(r"\((?:[^()]*?\b(?:recommendation|evidence)\b[^()]*)\)\s*\.?\s*$", re.IGNORECASE)
_NON_ALPHANUMERIC = re.compile(r"[^0-9a-z]")

_COLUMNS = ("doi", "ordinal_a", "ordinal_b", "biolord", "lexical", "verdict", "reason", "model", "model_digest",
            "text_a", "text_b")
_FLOAT_DIGITS = 4


class Verdict(str, Enum):
    DUPLICATE = "duplicate"  # same population, intervention, comparator and direction, restated
    DISTINCT = "distinct"


class DedupVerdictError(ValueError):
    """The frozen verdicts do not cover exactly the screened pairs; the package refuses to render."""

    def __init__(self, problems: list[str]):
        super().__init__("Duplicate verdicts out of step with the screen:\n" + "\n".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class CandidatePair:
    doi: str
    ordinal_a: int  # ordinal_a < ordinal_b
    ordinal_b: int
    biolord: float
    lexical: float
    text_a: str
    text_b: str


@dataclass(frozen=True)
class Adjudication:
    doi: str
    ordinal_a: int
    ordinal_b: int
    verdict: Verdict
    reason: str
    model: str = ""


@dataclass(frozen=True)
class RemovedRow:
    """One row dropped as a copy; duplicates.csv in the package."""
    doi: str
    society: str
    kept_ordinal: int
    removed_ordinal: int
    same_grade: bool
    kept_text: str
    removed_text: str


@dataclass(frozen=True)
class DedupResult:
    kept: list[HarmonizedRow]               # input order, one row per duplicate cluster
    removed: dict[tuple[str, int], int]     # (doi, removed ordinal) → kept ordinal
    n_grade_conflicts: int                  # clusters whose copies differ in harmonized grade
    removed_by_society: dict[str, int]


# ── screen ──────────────────────────────────────────────────────

def lexical_ratio(a: str, b: str) -> float:
    """Character similarity of the normalized texts; PDF noise ("inhi- bitor", "ﬁ", trailing grade) → 1.0."""
    return difflib.SequenceMatcher(None, _normalize(a), _normalize(b), autojunk=False).ratio()


def screen(doi: str, ordinals: Sequence[int], texts: Sequence[str], sim: np.ndarray) -> list[CandidatePair]:
    """Pairs of one guideline's rows at or above either screen threshold; sim is the BioLORD cosine matrix."""
    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            biolord = float(sim[i, j])
            lexical = lexical_ratio(texts[i], texts[j])
            if biolord < SCREEN_BIOLORD and lexical < SCREEN_LEXICAL:
                continue

            (oa, ta), (ob, tb) = sorted([(ordinals[i], texts[i]), (ordinals[j], texts[j])])
            pairs.append(CandidatePair(doi, oa, ob, round(biolord, _FLOAT_DIGITS), round(lexical, _FLOAT_DIGITS),
                                       ta, tb))
    return sorted(pairs, key=lambda p: (p.ordinal_a, p.ordinal_b))


def _normalize(text: str) -> str:
    text = _GRADE_TAIL.sub("", unicodedata.normalize("NFKC", text))
    return _NON_ALPHANUMERIC.sub("", text.lower())


# ── verdict file ────────────────────────────────────────────────

def write_adjudications(path: str, rows: Sequence[tuple[CandidatePair, Verdict, str, str, str]]) -> None:
    """rows: (pair, verdict, reason, model, model digest), sorted by doi then ordinals."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(_COLUMNS)
        for p, verdict, reason, model, digest in sorted(rows, key=lambda r: (r[0].doi, r[0].ordinal_a,
                                                                           r[0].ordinal_b)):
            w.writerow([p.doi, p.ordinal_a, p.ordinal_b, p.biolord, p.lexical, verdict.value, reason, model, digest,
                        p.text_a, p.text_b])


def load_adjudications(path: str) -> list[Adjudication]:
    with open(path, newline="", encoding="utf-8") as f:
        return [Adjudication(r["doi"], int(r["ordinal_a"]), int(r["ordinal_b"]), Verdict(r["verdict"]), r["reason"],
                             r["model"])
                for r in csv.DictReader(f)]


def check_complete(pairs: Sequence[CandidatePair], adjudications: Sequence[Adjudication]) -> None:
    """Exactly one verdict per screened pair; a verdict for an unscreened pair means another run or screen."""
    screened = {(p.doi, p.ordinal_a, p.ordinal_b) for p in pairs}
    judged = Counter((a.doi, a.ordinal_a, a.ordinal_b) for a in adjudications)
    problems = [f"{d} rows {a}, {b}: no verdict" for d, a, b in sorted(screened - set(judged))]
    problems += [f"{d} rows {a}, {b}: not screened" for d, a, b in sorted(set(judged) - screened)]
    problems += [f"{d} rows {a}, {b}: {n} verdicts" for (d, a, b), n in sorted(judged.items()) if n > 1]
    if problems:
        raise DedupVerdictError(problems)


# ── clusters ────────────────────────────────────────────────────

def deduplicate(rows: Sequence[HarmonizedRow], adjudications: Sequence[Adjudication]) -> DedupResult:
    """Duplicate verdicts → clusters (connected components) per guideline; one kept row each.

    Kept: both strength and certainty mapped first, then the lowest ordinal (e.g. the summary list).
    """
    by_key = {(r.doi, r.ordinal): r for r in rows}
    parent = {key: key for key in by_key}

    # Union-find over duplicate pairs, e.g. 0–1 and 1–2 → one cluster {0, 1, 2}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for a in adjudications:
        if a.verdict != Verdict.DUPLICATE:
            continue
        parent[find((a.doi, a.ordinal_b))] = find((a.doi, a.ordinal_a))

    clusters: dict[tuple, list[HarmonizedRow]] = {}
    for key, row in by_key.items():
        clusters.setdefault(find(key), []).append(row)

    removed = {}
    n_conflicts = 0
    removed_by_society = Counter()
    for members in clusters.values():
        if len(members) == 1:
            continue

        keep = min(members, key=lambda r: (not _fully_mapped(r), r.ordinal))
        n_conflicts += len({r.grade for r in members}) > 1
        for r in members:
            if r is keep:
                continue
            removed[(r.doi, r.ordinal)] = keep.ordinal
            removed_by_society[r.society] += 1

    kept = [r for r in rows if (r.doi, r.ordinal) not in removed]
    return DedupResult(kept, dict(sorted(removed.items())), n_conflicts, dict(sorted(removed_by_society.items())))


def removed_rows(rows: Sequence[HarmonizedRow], result: DedupResult) -> list[RemovedRow]:
    """Each removed row beside the copy kept in its place, sorted by doi and removed ordinal."""
    by_key = {(r.doi, r.ordinal): r for r in rows}
    out = []
    for (doi, ordinal), kept_ordinal in result.removed.items():
        removed, kept = by_key[(doi, ordinal)], by_key[(doi, kept_ordinal)]
        out.append(RemovedRow(doi, removed.society, kept_ordinal, ordinal, removed.grade == kept.grade,
                              kept.text, removed.text))
    return out


def remap_ordinals(doi: str, ordinals: Sequence[int], removed: dict[tuple[str, int], int]) -> tuple[int, ...]:
    """Ordinals that point to a removed copy point to the kept one instead, e.g. (3, 1) with 3 → 0 gives (0, 1)."""
    return tuple(sorted({removed.get((doi, o), o) for o in ordinals}))


def _fully_mapped(r: HarmonizedRow) -> bool:
    g = r.grade
    return g.category == Category.GRADED and g.strength_status == g.certainty_status == AxisStatus.MAPPED
