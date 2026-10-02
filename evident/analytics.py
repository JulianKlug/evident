"""Headline aggregates over harmonized rows of one published snapshot.

    [HarmonizedRow] ──headline──▶ [HeadlineRow] (one per society + ALL) ──write_headline──▶ CSV

Denominator rule: a % uses only graded rows whose axis is MAPPED; unmapped and
ungraded certainty are excluded and reported as counts. Empty denominator → empty cell.
"""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass, fields
from typing import Optional

from evident.domain import AxisStatus, Category, Certainty, Direction, HarmonizedGrade, Strength

_ALL = "ALL"
_PCT_DIGITS = 1
_HIGH_MODERATE = {Certainty.HIGH, Certainty.MODERATE}


@dataclass(frozen=True)
class HarmonizedRow:
    doi: str
    society: str
    year: int
    grade: HarmonizedGrade


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
    societies = sorted({r.society for r in rows})
    groups = [(s, [r for r in rows if r.society == s]) for s in societies] + [(_ALL, rows)]
    return [_aggregate(name, group, snapshot_id, extractor_version_id, model) for name, group in groups]


def write_headline(rows: list[HeadlineRow], path: str) -> None:
    """Atomic: written to <path>.tmp, then renamed, so a reader never sees half a file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(HeadlineRow)])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if v is None else v for k, v in asdict(row).items()})
    os.replace(tmp, path)


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


def _pct(num: int, den: int) -> Optional[float]:
    return round(100.0 * num / den, _PCT_DIGITS) if den else None
