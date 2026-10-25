"""Certainty of evidence per recommendation theme, from frozen theme assignments.

    corpus/themes/taxonomy.csv       reviewed theme list (LLM-drafted from rec topics, clinician-edited)
    corpus/themes/topic_themes.csv   guideline topic_id → theme, where a guideline covers one condition (ACP)
        │ scripts/assign_themes.py   LLM assigns every other rec to one taxonomy theme, from its text
        ▼
    corpus/themes/assignments.csv    rec_key(doi, text) → theme; frozen, read at publish
        │ theme_certainty(rows, assignments)
        ▼
    ThemeRow per (specialty, theme, certainty level)   % of the theme's graded recs with certainty MAPPED

Exploratory: themes are not prespecified, so they stay out of the paper package.
Denominator rule as in analytics: unmapped and ungraded certainty are not counted.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Optional

from evident.analytics import Edition, HarmonizedRow
from evident.domain import AxisStatus, Category, Certainty

# One clinical specialty per society; an unknown society is its own specialty
_SPECIALTIES = {"ACP": "Internal medicine", "ESICM": "Critical care", "SCCM": "Critical care"}

_CERTAINTY_ORDER = (Certainty.HIGH, Certainty.MODERATE, Certainty.LOW, Certainty.VERY_LOW)
_PCT_DIGITS = 1
_KEY_HEX = 16

OTHER = "Other"                  # the taxonomy's own catch-all
UNASSIGNED = "Not yet assigned"  # rec absent from the assignments file, e.g. extracted after it was frozen
_LAST = (OTHER, UNASSIGNED)      # drawn after every named theme, in this order


@dataclass(frozen=True)
class ThemeRow:
    specialty: str
    societies: tuple[str, ...]  # of the specialty's recs, e.g. ("ESICM", "SCCM")
    theme: int    # 1-based within the specialty, by mean certainty, highest first; Other, then unassigned, last
    label: str    # theme name, e.g. "Fluids"
    n_recs: int   # current recs in the theme, graded or not
    n_guidelines: int
    certainty: Certainty
    n: int
    pct: Optional[float]  # of the theme's graded recs with certainty MAPPED


def rec_key(doi: str, text: str) -> str:
    """Stable id of one recommendation across re-publishes: its guideline and its exact text."""
    digest = hashlib.sha256(f"{doi.lower()}\n{text}".encode("utf-8")).hexdigest()
    return digest[:_KEY_HEX]


def specialty(society: str) -> str:
    return _SPECIALTIES.get(society, society)


def theme_certainty(rows: list[HarmonizedRow], assignments: Mapping[str, str]) -> list[ThemeRow]:
    """Certainty distribution per theme and specialty, current editions only.

    assignments: rec_key → theme name. A rec without one is counted under UNASSIGNED, never dropped.
    """
    current = [r for r in rows if r.edition == Edition.CURRENT]
    out = []
    for name in sorted({specialty(r.society) for r in current}):
        members = [r for r in current if specialty(r.society) == name]
        societies = tuple(sorted({r.society for r in members}))
        themes: dict[str, list[HarmonizedRow]] = {}
        for r in members:
            themes.setdefault(assignments.get(rec_key(r.doi, r.text), UNASSIGNED), []).append(r)

        # Highest certainty first (mean rank, high = 0 … very low = 3): the heatmap reads as a gradient
        ranked = sorted(themes.items(), key=lambda t: (_LAST.index(t[0]) + 1 if t[0] in _LAST else 0,
                                                       _mean_rank(t[1]), -len(t[1]), t[0]))
        for i, (label, theme) in enumerate(ranked, start=1):
            out.extend(_rows(name, societies, i, label, theme))
    return out


def _rows(name: str, societies: tuple[str, ...], theme: int, label: str,
          members: list[HarmonizedRow]) -> list[ThemeRow]:
    mapped = _certainty_mapped(members)
    n_guidelines = len({r.doi for r in members})
    out = []
    for level in _CERTAINTY_ORDER:
        n = sum(c == level for c in mapped)
        out.append(ThemeRow(name, societies, theme, label, len(members), n_guidelines, level, n,
                            _pct(n, len(mapped))))
    return out


def _mean_rank(members: list[HarmonizedRow]) -> float:
    """Mean position in _CERTAINTY_ORDER; a theme without certainty-mapped recs goes last."""
    mapped = _certainty_mapped(members)
    return sum(_CERTAINTY_ORDER.index(c) for c in mapped) / len(mapped) if mapped else len(_CERTAINTY_ORDER)


def _certainty_mapped(members: list[HarmonizedRow]) -> list[Certainty]:
    return [r.grade.certainty for r in members
            if r.grade.category == Category.GRADED and r.grade.certainty_status == AxisStatus.MAPPED]


def _pct(num: int, den: int) -> Optional[float]:
    return round(100.0 * num / den, _PCT_DIGITS) if den else None
