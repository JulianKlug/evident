"""Grading crosswalk: native grade labels of any grading system → the GRADE-anchored harmonized vocabulary. No I/O
beyond reading the committed CSVs once.

    evident/grading_data/
        scales.csv      one row per native scale (S_… strength, E_… evidence)
        crosswalk.csv   one row per native code: harmonised value, fixed direction, sensitivity alternative
        families.csv    GradingFamily → default (strength scale, evidence scale)
        decisions.csv   methodological decisions D1–D14 behind the mappings

A guideline's raw labels are matched against its two scales, e.g. ESC:
    "Class I, Level C"   → strength code "I" (S_ESC_CLASS), evidence code "C" (E_ESC)
    raw_strength="[I, A]" (IDSA/ESMO, one field) → strength "A", evidence "I"
    "GRADE 1C"           → "1" + "C" (digit-letter split when "1c" itself is no code)

Matching works on token windows of the normalized label; the longest window wins and an exact match beats
one found after dropping filler words ("class", "level", "grade", …). Ties between different codes → no match.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Optional

from evident.domain import Direction, GradingFamily, ScalePair

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grading_data")
SCALE_COLUMNS = ("scale_key", "axis", "description", "equivalence", "codes_from")
CROSSWALK_COLUMNS = ("scale_key", "axis", "native_code", "aliases", "native_meaning", "harmonised", "direction",
                     "alternative", "confidence", "note")
FAMILY_COLUMNS = ("family", "strength_scale", "evidence_scale")
DECISION_COLUMNS = ("id", "question", "rule", "alternative", "rationale")
WILDCARD = "*"  # the scale has no native labels: every statement gets this row (e.g. S_NONE, S_NO_GRADING)

_ALIAS_SEPARATOR = "|"
_MAX_WINDOW = 6
_STRENGTH_AXIS, _EVIDENCE_AXIS = "strength", "evidence"
_OWN_CODES, _STRENGTH_GRADE_CODES = "own", "strength_grade"  # scales.csv codes_from

# Filler words around a code, e.g. "Strength of recommendation B" → "b", "LoE A-/RG 1-" → "a-", "1-"
_FILLER_WORDS = {"class", "level", "levels", "grade", "grades", "loe", "lor", "cor", "coe", "of", "evidence",
                 "recommendation", "strength", "quality", "certainty", "category", "cat", "ce", "rg", "type", "the"}
_MINUS_SIGNS = {"−": "-", "–": "-", "‐": "-"}
_DIGIT_LETTER = re.compile(r"^(\d)([a-d])$")  # GRADE "1c" → "1", "c"
# Panel agreement is never strength (D8): "strong consensus" must not read as "strong"
_CONSENSUS_PATTERN = r"\b(?:very\s+|strong(?:est)?\s+|broad\s+)?consensus\b|\bagreement\b|\d+(?:\.\d+)?\s*%"


class Harmonised(str, Enum):
    """Workbook vocabulary (GRADE anchor); evident.harmonization translates it to the domain enums."""
    STRONG = "STRONG"
    CONDITIONAL = "CONDITIONAL"
    NO_REC = "NO_REC"
    UNGRADED = "UNGRADED"
    NOT_REPORTED = "NOT_REPORTED"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"
    NOT_APPLICABLE = "NOT_APPLICABLE"


_STRENGTH_VALUES = {Harmonised.STRONG, Harmonised.CONDITIONAL, Harmonised.NO_REC, Harmonised.UNGRADED,
                    Harmonised.NOT_REPORTED}
_EVIDENCE_VALUES = {Harmonised.HIGH, Harmonised.MODERATE, Harmonised.LOW, Harmonised.VERY_LOW,
                    Harmonised.NOT_APPLICABLE, Harmonised.NOT_REPORTED}
_FIXED_DIRECTIONS = {"FOR": Direction.FOR, "AGAINST": Direction.AGAINST}
_FREE_DIRECTIONS = {"", "as stated", "NEUTRAL"}  # NEUTRAL only occurs with NO_REC / NOT_REPORTED


class GradingTableError(ValueError):
    """The committed CSVs are inconsistent; raised on first load."""


@dataclass(frozen=True)
class CrosswalkEntry:
    scale: str
    native_code: str
    value: Harmonised
    alternative: Optional[Harmonised]
    direction: Optional[Direction]  # set only when the grade itself fixes it, e.g. ESC III → AGAINST
    native_meaning: str
    confidence: str
    note: str


@dataclass(frozen=True)
class _Scale:
    key: str
    axis: str
    description: str
    entries: tuple
    exact: dict      # normalized code or alias → entry
    stripped: dict   # same, filler words dropped; keys shared by different mappings removed
    wildcard: Optional[CrosswalkEntry]
    # Evidence read off the strength grade itself, e.g. EPA prints only "B": certainty from B's definition
    codes_from_strength: bool = False


@dataclass(frozen=True)
class _Tables:
    scales: dict
    families: dict   # GradingFamily → ScalePair (or None for OTHER)
    decisions: tuple


# ---------------------------------------------------------------- public API

def resolve_scales(family: GradingFamily, override: Optional[ScalePair]) -> Optional[ScalePair]:
    """The guideline's own scales (manifest) win over the family default; None when neither exists."""
    if override is not None and override.strength and override.evidence:
        return override
    return _tables().families.get(family)


def match(scales: ScalePair, raw_strength: str, raw_certainty: str
          ) -> tuple[Optional[CrosswalkEntry], Optional[CrosswalkEntry]]:
    """(strength entry, evidence entry) for the raw labels; None where no code is found.

    Each axis searches its own field first, then the other field, e.g. ESMO "[I, A]" extracted into
    raw_strength only. In the other field, windows that are also codes of the other axis are skipped.
    """
    tables = _tables()
    strength_scale = _scale(tables, scales.strength, _STRENGTH_AXIS)
    evidence_scale = _scale(tables, scales.evidence, _EVIDENCE_AXIS)

    strength = _find(strength_scale, raw_strength) or _find(strength_scale, raw_certainty, evidence_scale)
    if evidence_scale.codes_from_strength:
        evidence = evidence_scale.exact.get(" ".join(_normalize(strength.native_code))) if strength else None
        return strength, evidence

    evidence = _find(evidence_scale, raw_certainty) or _find(evidence_scale, raw_strength, strength_scale)
    return strength, evidence


def entries(scale_key: str) -> tuple:
    return _scale(_tables(), scale_key, None).entries


def render_doc() -> str:
    """docs/harmonization_tables.md, generated so the published mapping cannot drift from the code."""
    tables = _tables()
    lines = ["# Harmonization tables", "",
             "<!-- Generated by `python scripts/export_grading_tables.py --doc` from evident/grading_data/. "
             "Do not edit. -->", "",
             "Native grades mapped onto the GRADE-anchored scale. *Alternative* is the value used in the "
             "sensitivity analysis; *direction* is set only where the grade itself fixes it.", "",
             "## Decisions", "", "| ID | Question | Rule | Alternative |", "|---|---|---|---|"]
    lines += [f"| {d['id']} | {_cell(d['question'])} | {_cell(d['rule'])} | {_cell(d['alternative'])} |"
              for d in tables.decisions]

    lines += ["", "## Families (default scales)", "", "| Family | Strength scale | Evidence scale |", "|---|---|---|"]
    for family, pair in sorted(tables.families.items(), key=lambda kv: kv[0].value):
        lines.append(f"| {family.value} | {pair.strength if pair else '—'} | {pair.evidence if pair else '—'} |")

    for scale in sorted(tables.scales.values(), key=lambda s: (s.axis != _STRENGTH_AXIS, s.key)):
        lines += ["", f"## {scale.key} ({scale.axis})", "", _cell(scale.description), "",
                  "| Native code | Meaning | Harmonised | Direction | Alternative | Confidence | Note |",
                  "|---|---|---|---|---|---|---|"]
        for e in scale.entries:
            lines.append(f"| {_cell(e.native_code)} | {_cell(e.native_meaning)} | {e.value.value} | "
                         f"{e.direction.value if e.direction else ''} | "
                         f"{e.alternative.value if e.alternative else ''} | {e.confidence} | {_cell(e.note)} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- matching

def _normalize(label: str) -> list[str]:
    """'Class IIa, Level B-NR.' → ['class', 'iia', 'level', 'b-nr']; arrows, +, -, % kept."""
    text = re.sub(_CONSENSUS_PATTERN, " ", str(label or "").lower())
    for sign, ascii_minus in _MINUS_SIGNS.items():
        text = text.replace(sign, ascii_minus)
    text = re.sub(r"[^0-9a-z+\-↑↓↔θ%.]+", " ", text)
    return [t.strip(".") for t in text.split() if t.strip(".")]


def _strip_filler(tokens: list[str]) -> str:
    return " ".join(t for t in tokens if t not in _FILLER_WORDS)


def _windows(tokens: list[str]):
    for size in range(min(_MAX_WINDOW, len(tokens)), 0, -1):
        for start in range(len(tokens) - size + 1):
            yield tokens[start:start + size]


def _lookup(scale: _Scale, window: list[str]):
    """(quality, entry) for one token window.

    quality = (content words matched, exact beats stripped, longer window), e.g. for "Strong recommendation"
    the code "Strong recommendation" (1 content word) beats "Recommendation" (0); "Level I" beats "I".
    """
    stripped = _strip_filler(window)
    content = len(stripped.split())
    exact = scale.exact.get(" ".join(window))
    if exact:
        return (content, 1, len(window)), exact
    if stripped and stripped in scale.stripped:
        return (content, 0, len(window)), scale.stripped[stripped]
    return None


def _best(scale: _Scale, tokens: list[str], other: Optional[_Scale]) -> Optional[CrosswalkEntry]:
    found = []
    for window in _windows(tokens):
        hit = _lookup(scale, window)
        if hit is None:
            continue
        if other is not None and _lookup(other, window) is not None:
            continue  # e.g. "a" in "[I, A]" is a code of the other axis
        found.append(hit)
    if not found:
        return None

    top = max(quality for quality, _ in found)
    winners = {entry for quality, entry in found if quality == top}
    if len(winners) == 1:
        return winners.pop()
    values = {(e.value, e.alternative, e.direction) for e in winners}
    return winners.pop() if len(values) == 1 else None  # equivalent codes, e.g. "A" and "Level I"


def _split_digit_letter(tokens: list[str]) -> list[str]:
    out = []
    for t in tokens:
        m = _DIGIT_LETTER.match(t)
        out += [m.group(1), m.group(2)] if m else [t]
    return out


def _find(scale: _Scale, label: str, other: Optional[_Scale] = None) -> Optional[CrosswalkEntry]:
    if scale.wildcard is not None:
        return scale.wildcard
    tokens = _normalize(label)
    if not tokens:
        return None
    return _best(scale, tokens, other) or _best(scale, _split_digit_letter(tokens), other)


# ---------------------------------------------------------------- loading

def _scale(tables: _Tables, key: str, axis: Optional[str]) -> _Scale:
    scale = tables.scales.get(key)
    if scale is None:
        raise GradingTableError(f"Unknown grading scale '{key}'")
    if axis is not None and scale.axis != axis:
        raise GradingTableError(f"Scale '{key}' is on the {scale.axis} axis, expected {axis}")
    return scale


def _read(name: str) -> list[dict]:
    with open(os.path.join(DATA_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _harmonised(value: str, allowed: set, where: str) -> Harmonised:
    try:
        parsed = Harmonised(value)
    except ValueError:
        raise GradingTableError(f"{where}: '{value}' is not a harmonised value") from None
    if parsed not in allowed:
        raise GradingTableError(f"{where}: '{value}' is not allowed on this axis")
    return parsed


def _entry(row: dict, axis: str) -> CrosswalkEntry:
    where = f"{row['scale_key']} '{row['native_code']}'"
    allowed = _STRENGTH_VALUES if axis == _STRENGTH_AXIS else _EVIDENCE_VALUES
    direction = row["direction"]
    if direction not in _FIXED_DIRECTIONS and direction not in _FREE_DIRECTIONS:
        raise GradingTableError(f"{where}: unknown direction '{direction}'")
    return CrosswalkEntry(
        scale=row["scale_key"], native_code=row["native_code"],
        value=_harmonised(row["harmonised"], allowed, where),
        alternative=_harmonised(row["alternative"], allowed, where) if row["alternative"] else None,
        direction=_FIXED_DIRECTIONS.get(direction),
        native_meaning=row["native_meaning"], confidence=row["confidence"], note=row["note"])


def _build_scale(key: str, axis: str, description: str, rows: list[dict], codes_from: str) -> _Scale:
    entries = tuple(_entry(r, axis) for r in rows)
    exact, stripped, conflicts, wildcard = {}, {}, set(), None

    for entry, row in zip(entries, rows):
        if entry.native_code == WILDCARD:
            wildcard = entry
            continue
        for code in [entry.native_code] + [a for a in row["aliases"].split(_ALIAS_SEPARATOR) if a]:
            tokens = _normalize(code)
            key_exact = " ".join(tokens)
            previous = exact.get(key_exact)
            if previous is not None and previous.value != entry.value:
                raise GradingTableError(f"{key}: '{code}' maps to both {previous.value} and {entry.value}")
            exact.setdefault(key_exact, entry)

            key_stripped = _strip_filler(tokens)
            previous = stripped.get(key_stripped)
            if previous is not None and previous.value != entry.value:
                conflicts.add(key_stripped)  # e.g. "Level I" (A) vs "I" (Insufficient): exact match only
            stripped.setdefault(key_stripped, entry)

    for k in conflicts:
        stripped.pop(k)
    stripped.pop("", None)
    return _Scale(key, axis, description, entries, exact, stripped, wildcard,
                  codes_from_strength=codes_from == _STRENGTH_GRADE_CODES)


@lru_cache(maxsize=1)
def _tables() -> _Tables:
    by_scale = {}
    for row in _read("crosswalk.csv"):
        by_scale.setdefault(row["scale_key"], []).append(row)

    scales = {}
    for row in _read("scales.csv"):
        key, axis = row["scale_key"], row["axis"]
        if axis not in (_STRENGTH_AXIS, _EVIDENCE_AXIS):
            raise GradingTableError(f"{key}: unknown axis '{axis}'")
        if key not in by_scale:
            raise GradingTableError(f"{key}: no crosswalk rows")
        codes_from = row["codes_from"]
        if codes_from not in (_OWN_CODES, _STRENGTH_GRADE_CODES):
            raise GradingTableError(f"{key}: unknown codes_from '{codes_from}'")
        if codes_from == _STRENGTH_GRADE_CODES and axis != _EVIDENCE_AXIS:
            raise GradingTableError(f"{key}: only an evidence scale can take its codes from the strength grade")
        scales[key] = _build_scale(key, axis, row["description"], by_scale.pop(key), codes_from)
    if by_scale:
        raise GradingTableError(f"Crosswalk rows for undeclared scales: {sorted(by_scale)}")

    families = {}
    for row in _read("families.csv"):
        family = GradingFamily(row["family"])
        if not row["strength_scale"]:
            families[family] = None
            continue
        _scale_of(scales, row["strength_scale"], _STRENGTH_AXIS)
        _scale_of(scales, row["evidence_scale"], _EVIDENCE_AXIS)
        families[family] = ScalePair(row["strength_scale"], row["evidence_scale"])

    return _Tables(scales, families, tuple(_read("decisions.csv")))


def _scale_of(scales: dict, key: str, axis: str) -> None:
    if key not in scales or scales[key].axis != axis:
        raise GradingTableError(f"Family scale '{key}' missing or not a {axis} scale")


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")
