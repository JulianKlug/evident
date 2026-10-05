"""Aggregates over harmonized rows of one published snapshot. Pure: rows in, tables out.

    [HarmonizedRow] (all editions) ─┬─ current ──▶ headline, certainty_distribution,
                                    │              strength_by_certainty, guideline_summary
                                    ├─ all ──────▶ trends (by society × year)
                                    └────────────▶ caption_facts (counts every caption states)

Denominator rule: a % uses only graded rows whose axis is MAPPED; unmapped and
ungraded certainty are excluded and reported as counts. Empty denominator → empty cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

from evident.domain import AxisStatus, Category, Certainty, Direction, HarmonizedGrade, Strength

_ALL = "ALL"
_PCT_DIGITS = 1
_HIGH_MODERATE = {Certainty.HIGH, Certainty.MODERATE}
_CERTAINTY_ORDER = (Certainty.HIGH, Certainty.MODERATE, Certainty.LOW, Certainty.VERY_LOW)
_STRENGTH_ORDER = (Strength.STRONG, Strength.WEAK)


class Edition(str, Enum):
    CURRENT = "current"        # latest edition of its topic: counted in the current map
    SUPERSEDED = "superseded"  # older edition: counted in trends only


@dataclass(frozen=True)
class HarmonizedRow:
    doi: str
    society: str
    year: int
    grade: HarmonizedGrade
    edition: Edition = Edition.CURRENT
    text: str = ""
    page: Optional[int] = None
    raw_strength: str = ""
    raw_certainty: str = ""


@dataclass(frozen=True)
class GuidelineFacts:
    """One manifest guideline as published; built by the pipeline from manifest + snapshot members."""
    doi: str
    society: str
    year: int
    title: str
    topic_id: str
    edition: Edition
    joint_with: tuple[str, ...] = ()
    excluded_reason: Optional[str] = None


@dataclass(frozen=True)
class CertaintyRow:
    society: str
    certainty: Certainty
    n: int
    pct: Optional[float]  # of graded recs with certainty MAPPED


@dataclass(frozen=True)
class CrosstabRow:
    society: str
    strength: Strength
    certainty: Certainty
    n: int
    pct_of_strength: Optional[float]  # row %: share of this strength's recs at this certainty


@dataclass(frozen=True)
class TrendRow:
    society: str
    year: int
    n_guidelines: int
    n_recs: int
    n_graded: int
    pct_strong: Optional[float]
    pct_high_moderate: Optional[float]


@dataclass(frozen=True)
class GuidelineRow:
    doi: str
    society: str
    year: int
    title: str
    edition: Edition
    joint_with: str  # ";"-joined, empty if none
    excluded_reason: str
    n_recs: int
    n_graded: int
    ungraded_share: Optional[float]
    pct_strong: Optional[float]
    pct_high_moderate: Optional[float]


@dataclass(frozen=True)
class CaptionFacts:
    """Counts every figure and table caption states (design OV8)."""
    societies: tuple[str, ...]
    n_guidelines: int                         # current editions with an active run
    n_recs: int
    n_graded: int
    n_superseded_guidelines: int              # older editions: in trends, not in the current map
    n_recs_all_editions: int
    n_strength_unmapped: int
    n_certainty_unmapped: int
    n_certainty_ungraded: int
    excluded: tuple[tuple[str, str], ...]     # (doi, reason)
    joint: tuple[tuple[str, str, tuple[str, ...]], ...]  # (doi, society, joint_with), current editions
    unvalidated_societies: tuple[str, ...]    # no guideline of theirs in the labelled set
    year_min: Optional[int]
    year_max: Optional[int]


@dataclass(frozen=True)
class HeadlineRow:
    society: str
    n_guidelines: int
    n_recs: int
    n_graded: int
    n_ungraded: int
    ungraded_share: Optional[float]
    pct_strong: Optional[float]
    pct_against: Optional[float]
    pct_high_moderate: Optional[float]
    n_strength_unmapped: int
    n_certainty_unmapped: int
    n_certainty_ungraded: int
    snapshot_id: int
    extractor_version_id: str
    model: str  # LLM the snapshot ran on, e.g. "qwen3:14b@bdbd181c33f2"


def headline(rows: list[HarmonizedRow], snapshot_id: int, extractor_version_id: str,
             model: str) -> list[HeadlineRow]:
    """Current editions only (design: "current map uses the latest edition")."""
    return [_aggregate(name, group, snapshot_id, extractor_version_id, model)
            for name, group in _by_society(_current(rows))]


def certainty_distribution(rows: list[HarmonizedRow]) -> list[CertaintyRow]:
    """Per society + ALL: count and % of each certainty level, over graded recs with certainty MAPPED."""
    out = []
    for society, group in _by_society(_current(rows)):
        mapped = [g for g in _graded(group) if g.certainty_status == AxisStatus.MAPPED]
        for level in _CERTAINTY_ORDER:
            n = sum(g.certainty == level for g in mapped)
            out.append(CertaintyRow(society, level, n, _pct(n, len(mapped))))
    return out


def strength_by_certainty(rows: list[HarmonizedRow]) -> list[CrosstabRow]:
    """Per society + ALL: strength × certainty counts, both axes MAPPED; row % within each strength.

    Example: 10 strong recs, 4 on low certainty → (STRONG, LOW, n=4, pct_of_strength=40.0).
    """
    out = []
    for society, group in _by_society(_current(rows)):
        both = [g for g in _graded(group)
                if g.strength_status == AxisStatus.MAPPED and g.certainty_status == AxisStatus.MAPPED]
        for strength in _STRENGTH_ORDER:
            of_strength = [g for g in both if g.strength == strength]
            for level in _CERTAINTY_ORDER:
                n = sum(g.certainty == level for g in of_strength)
                out.append(CrosstabRow(society, strength, level, n, _pct(n, len(of_strength))))
    return out


def trends(rows: list[HarmonizedRow]) -> list[TrendRow]:
    """Per society + ALL and publication year, over all editions (superseded ones included)."""
    out = []
    for society, group in _by_society(rows):
        for year in sorted({r.year for r in group}):
            in_year = [r for r in group if r.year == year]
            graded = _graded(in_year)
            strength_mapped = [g for g in graded if g.strength_status == AxisStatus.MAPPED]
            certainty_mapped = [g for g in graded if g.certainty_status == AxisStatus.MAPPED]
            out.append(TrendRow(
                society=society, year=year,
                n_guidelines=len({r.doi for r in in_year}),
                n_recs=len(in_year),
                n_graded=len(graded),
                pct_strong=_pct(sum(g.strength == Strength.STRONG for g in strength_mapped), len(strength_mapped)),
                pct_high_moderate=_pct(sum(g.certainty in _HIGH_MODERATE for g in certainty_mapped),
                                       len(certainty_mapped)),
            ))
    return out


def guideline_summary(rows: list[HarmonizedRow], guidelines: Sequence[GuidelineFacts]) -> list[GuidelineRow]:
    """One row per manifest guideline, sorted by society, year, doi; excluded ones have 0 recs."""
    by_doi: dict[str, list[HarmonizedRow]] = {}
    for r in rows:
        by_doi.setdefault(r.doi, []).append(r)

    out = []
    for g in sorted(guidelines, key=lambda f: (f.society, f.year, f.doi)):
        grades = [r.grade for r in by_doi.get(g.doi, [])]
        graded = [x for x in grades if x.category == Category.GRADED]
        strength_mapped = [x for x in graded if x.strength_status == AxisStatus.MAPPED]
        certainty_mapped = [x for x in graded if x.certainty_status == AxisStatus.MAPPED]
        out.append(GuidelineRow(
            doi=g.doi, society=g.society, year=g.year, title=g.title, edition=g.edition,
            joint_with=";".join(g.joint_with), excluded_reason=g.excluded_reason or "",
            n_recs=len(grades), n_graded=len(graded),
            ungraded_share=_pct(len(grades) - len(graded), len(grades)),
            pct_strong=_pct(sum(x.strength == Strength.STRONG for x in strength_mapped), len(strength_mapped)),
            pct_high_moderate=_pct(sum(x.certainty in _HIGH_MODERATE for x in certainty_mapped),
                                   len(certainty_mapped)),
        ))
    return out


def caption_facts(rows: list[HarmonizedRow], guidelines: Sequence[GuidelineFacts],
                  labelled_dois: set[str]) -> CaptionFacts:
    """Unvalidated societies are derived, not hard-coded: no labelled guideline in the corpus."""
    current = _current(rows)
    graded = _graded(current)
    societies = tuple(sorted({g.society for g in guidelines}))
    validated = {g.society for g in guidelines if g.doi in labelled_dois}
    current_dois = {r.doi for r in current}
    years = [r.year for r in rows]

    return CaptionFacts(
        societies=societies,
        n_guidelines=len(current_dois),
        n_recs=len(current),
        n_graded=len(graded),
        n_superseded_guidelines=len({r.doi for r in rows if r.edition == Edition.SUPERSEDED}),
        n_recs_all_editions=len(rows),
        n_strength_unmapped=sum(g.strength_status == AxisStatus.UNMAPPED for g in graded),
        n_certainty_unmapped=sum(g.certainty_status == AxisStatus.UNMAPPED for g in graded),
        n_certainty_ungraded=sum(g.certainty_status == AxisStatus.UNGRADED for g in graded),
        excluded=tuple(sorted((g.doi, g.excluded_reason) for g in guidelines if g.excluded_reason)),
        joint=tuple(sorted((g.doi, g.society, g.joint_with) for g in guidelines
                           if g.joint_with and g.doi in current_dois)),
        unvalidated_societies=tuple(s for s in societies if s not in validated),
        year_min=min(years) if years else None,
        year_max=max(years) if years else None,
    )


def _aggregate(society: str, rows: list[HarmonizedRow], snapshot_id: int, version_id: str,
               model: str) -> HeadlineRow:
    grades = [r.grade for r in rows]
    graded = [g for g in grades if g.category == Category.GRADED]
    strength_mapped = [g for g in graded if g.strength_status == AxisStatus.MAPPED]
    certainty_mapped = [g for g in graded if g.certainty_status == AxisStatus.MAPPED]

    return HeadlineRow(
        society=society,
        n_guidelines=len({r.doi for r in rows}),
        n_recs=len(grades),
        n_graded=len(graded),
        n_ungraded=len(grades) - len(graded),
        ungraded_share=_pct(len(grades) - len(graded), len(grades)),
        pct_strong=_pct(sum(g.strength == Strength.STRONG for g in strength_mapped), len(strength_mapped)),
        pct_against=_pct(sum(g.direction == Direction.AGAINST for g in strength_mapped), len(strength_mapped)),
        pct_high_moderate=_pct(sum(g.certainty in _HIGH_MODERATE for g in certainty_mapped), len(certainty_mapped)),
        n_strength_unmapped=sum(g.strength_status == AxisStatus.UNMAPPED for g in graded),
        n_certainty_unmapped=sum(g.certainty_status == AxisStatus.UNMAPPED for g in graded),
        n_certainty_ungraded=sum(g.certainty_status == AxisStatus.UNGRADED for g in graded),
        snapshot_id=snapshot_id,
        extractor_version_id=version_id,
        model=model,
    )


def _current(rows: list[HarmonizedRow]) -> list[HarmonizedRow]:
    return [r for r in rows if r.edition == Edition.CURRENT]


def _graded(rows: list[HarmonizedRow]) -> list[HarmonizedGrade]:
    return [r.grade for r in rows if r.grade.category == Category.GRADED]


def _by_society(rows: list[HarmonizedRow]) -> list[tuple[str, list[HarmonizedRow]]]:
    """[(society, its rows)] sorted by name, then (ALL, every row)."""
    societies = sorted({r.society for r in rows})
    return [(s, [r for r in rows if r.society == s]) for s in societies] + [(_ALL, rows)]


def _pct(num: int, den: int) -> Optional[float]:
    return round(100.0 * num / den, _PCT_DIGITS) if den else None
