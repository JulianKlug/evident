"""Raw extracted grade + text → HarmonizedGrade. Pure, no I/O.

Rules run in order: clean → category → (ungraded? stop) → strength → direction → certainty.
Examples (GRADE):
    "strong recommendation;", "high-certainty evidence", "ACP recommends against …"
        → STRONG / AGAINST / HIGH
    "onditional recommendation", "moderate/low-certainty evidence", "ACP suggests …"
        → WEAK / FOR / LOW   (truncation repaired, composite takes the lower)
"""

from __future__ import annotations

import re
from typing import Optional

from evident.domain import (
    AxisStatus,
    Category,
    Certainty,
    Direction,
    GradingFamily,
    HarmonizedGrade,
    Strength,
)

_AGAINST_PATTERNS = (r"\b(recommends?|suggests?)\s+against\b",
                     r"\b(recommends?|suggests?)\s+not\b",
                     r"\bshould\s+not\b", r"\bdo\s+not\s+(use|give|administer|perform)\b")
_NO_REC_PATTERNS = (r"\bcannot\s+recommend\b", r"\bno\s+recommendation\b",
                    r"\bevidence\s+(was|is)\s+(inconclusive|insufficient)\b")
_BEST_PRACTICE_PATTERN = r"best practice|good practice statement"

_PLACEHOLDERS = {"0", "0.0", "nan", "na", "n/a", ""}
_NO_RECOMMENDATION_LABEL = "no recommendation"
_MIN_REPAIR_LENGTH = 4

_STRENGTH_VOCAB = {"strong": Strength.STRONG, "conditional": Strength.WEAK, "weak": Strength.WEAK}
_DIRECTION_VOCAB = {"for": Direction.FOR, "against": Direction.AGAINST}

# GRADE labels are often hyphenated ("high-certainty"), so both forms are vocabulary
_CERTAINTY_LEVELS = {"high": Certainty.HIGH, "moderate": Certainty.MODERATE,
                     "low": Certainty.LOW, "very low": Certainty.VERY_LOW}
_CERTAINTY_WORD_FORMS = {"high": "high", "high-certainty": "high",
                         "moderate": "moderate", "moderate-certainty": "moderate",
                         "low": "low", "low-certainty": "low",
                         "very": "very", "very-low": "very low"}
_CERTAINTY_STOPWORDS = {"certainty", "of", "evidence", "quality"}
_CERTAINTY_STOP_PHRASE = "confidence in estimates of effect"
_INSUFFICIENT = "insufficient"
_CERTAINTY_ORDER = [Certainty.VERY_LOW, Certainty.LOW, Certainty.MODERATE, Certainty.HIGH]


class UnsupportedGradingFamilyError(ValueError):
    """Only GRADE is harmonized in M0; other families arrive in M6."""


def harmonize(raw_strength: str, raw_certainty: str, text: str,
              raw_category: Category, family: GradingFamily) -> HarmonizedGrade:
    if family != GradingFamily.GRADE:
        raise UnsupportedGradingFamilyError(f"No harmonization for grading family '{family}'")

    strength_label = _clean(raw_strength)
    certainty_label = _clean(raw_certainty)
    category = _category(strength_label, text, Category(raw_category))

    # Ungraded statements carry no grade axes at all
    if category != Category.GRADED:
        return HarmonizedGrade(category, None, None, None, AxisStatus.UNGRADED, AxisStatus.UNGRADED)

    strength = _strength(strength_label)
    direction = _direction(strength_label, text) if strength else None
    certainty, certainty_status = _certainty(certainty_label)

    return HarmonizedGrade(
        category=category,
        strength=strength,
        direction=direction,
        certainty=certainty,
        strength_status=AxisStatus.MAPPED if strength else AxisStatus.UNMAPPED,
        certainty_status=certainty_status,
    )


def _clean(value: str) -> str:
    """' Strong  Recommendation; ' → 'strong recommendation'; placeholders → ''."""
    cleaned = re.sub(r"\s+", " ", str(value).lower().strip()).rstrip(";,.").strip()
    return "" if cleaned in _PLACEHOLDERS else cleaned


def _matches_any(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _category(strength_label: str, text: str, raw_category: Category) -> Category:
    if raw_category != Category.GRADED:
        return raw_category
    if strength_label == _NO_RECOMMENDATION_LABEL or _matches_any(_NO_REC_PATTERNS, text):
        return Category.NO_RECOMMENDATION
    if re.search(_BEST_PRACTICE_PATTERN, f"{strength_label} {text}", re.IGNORECASE):
        return Category.BEST_PRACTICE
    return Category.GRADED


def _repair(token: str, vocabulary) -> Optional[str]:
    """Exact vocabulary word, or the single word the token is a proper suffix of.

    'trong' → 'strong'; 'ertainty' (suffix of several) → None.
    """
    if token in vocabulary:
        return token
    if len(token) < _MIN_REPAIR_LENGTH:
        return None
    candidates = [w for w in vocabulary if w.endswith(token) and w != token]
    return candidates[0] if len(candidates) == 1 else None


def _label_tokens(label: str) -> list[str]:
    return [t for t in re.split(r"[\s,;]+", label) if t]


def _strength(label: str) -> Optional[Strength]:
    found = {_STRENGTH_VOCAB[w] for w in (_repair(t, _STRENGTH_VOCAB) for t in _label_tokens(label)) if w}
    return found.pop() if len(found) == 1 else None


def _direction(label: str, text: str) -> Direction:
    # Label wins over text, e.g. "Strong For" on "suggest against" text stays FOR
    for token in _label_tokens(label):
        if token in _DIRECTION_VOCAB:
            return _DIRECTION_VOCAB[token]
    return Direction.AGAINST if _matches_any(_AGAINST_PATTERNS, text) else Direction.FOR


def _certainty(label: str) -> tuple[Optional[Certainty], AxisStatus]:
    label = label.replace(_CERTAINTY_STOP_PHRASE, " ").strip()
    if not label:
        return None, AxisStatus.UNMAPPED

    # Composite "moderate/low" → each part, then the lower one
    levels = []
    for part in label.split("/"):
        tokens = [t for t in part.split() if t not in _CERTAINTY_STOPWORDS]
        if tokens == [_INSUFFICIENT]:
            return None, AxisStatus.UNGRADED

        words = [_repair(t, _CERTAINTY_WORD_FORMS) for t in tokens]
        if not words or None in words:
            return None, AxisStatus.UNMAPPED

        level = _CERTAINTY_LEVELS.get(" ".join(_CERTAINTY_WORD_FORMS[w] for w in words))
        if level is None:
            return None, AxisStatus.UNMAPPED
        levels.append(level)

    return min(levels, key=_CERTAINTY_ORDER.index), AxisStatus.MAPPED
