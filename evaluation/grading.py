from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GradingScheme:
    name: str
    grades: list[str]
    levels: list[str]
    grade_aliases: dict[str, str] = field(default_factory=dict)
    level_aliases: dict[str, str] = field(default_factory=dict)

    def normalize_grade(self, raw: str) -> str | None:
        """Normalize a grade string to its canonical form, or None if not recognized."""
        if raw is None:
            return None
        key = raw.strip().lower()
        # Check aliases first
        if key in self.grade_aliases:
            return self.grade_aliases[key]
        # Check canonical values (case-insensitive)
        for g in self.grades:
            if g.lower() == key:
                return g
        return None

    def normalize_level(self, raw: str) -> str | None:
        """Normalize a level string to its canonical form, or None if not recognized."""
        if raw is None:
            return None
        key = raw.strip().lower()
        # Check aliases first
        if key in self.level_aliases:
            return self.level_aliases[key]
        # Check canonical values (case-insensitive)
        for lev in self.levels:
            if lev.lower() == key:
                return lev
        return None


ESC_ERS = GradingScheme(
    name="esc_ers",
    grades=["I", "IIa", "IIb", "III"],
    levels=["A", "B", "C"],
    grade_aliases={
        "1": "I",
        "2a": "IIa",
        "iia": "IIa",
        "2b": "IIb",
        "iib": "IIb",
        "3": "III",
        "iii": "III",
        "class i": "I",
        "class ii a": "IIa",
        "class ii b": "IIb",
        "class iii": "III",
        "class iia": "IIa",
        "class iib": "IIb",
        "class 1": "I",
        "class 2a": "IIa",
        "class 2b": "IIb",
        "class 3": "III",
        "ii a": "IIa",
        "ii b": "IIb",
    },
    level_aliases={
        "level a": "A",
        "level b": "B",
        "level c": "C",
        "level of evidence a": "A",
        "level of evidence b": "B",
        "level of evidence c": "C",
    },
)

ABCD_123 = GradingScheme(
    name="abcd_123",
    grades=["A", "B", "C", "D"],
    levels=["1", "2", "3"],
    grade_aliases={
        "grade a": "A",
        "grade b": "B",
        "grade c": "C",
        "grade d": "D",
        "a (highest)": "A",
        "b (moderate)": "B",
    },
    level_aliases={
        "level 1": "1",
        "level 2": "2",
        "level 3": "3",
        "level of evidence 1": "1",
        "level of evidence 2": "2",
        "level of evidence 3": "3",
        "i": "1",
        "ii": "2",
        "iii": "3",
        "iia": "2",
        "iit": "2",
        "++": "1",
        "+": "2",
    },
)

GRADE = GradingScheme(
    name="grade",
    grades=["Strong For", "Weak For", "Weak Against", "Strong Against"],
    levels=["High", "Moderate", "Low", "Very Low"],
    grade_aliases={
        "strong for": "Strong For",
        "weak for": "Weak For",
        "conditional for": "Weak For",
        "weak against": "Weak Against",
        "conditional against": "Weak Against",
        "strong against": "Strong Against",
        "strong": "Strong For",
        "strong recommendation": "Strong For",
        "strong recommendation for": "Strong For",
        "strong recommendation against": "Strong Against",
        "conditional recommendation": "Weak For",
        "conditional recommendation for": "Weak For",
        "conditional recommendation against": "Weak Against",
        "conditional": "Weak For",
        "weak recommendation": "Weak For",
        "weak recommendation for": "Weak For",
        "weak recommendation against": "Weak Against",
        # Truncation artifacts in GT data
        "onditional recommendation": "Weak For",
        "trong recommendation": "Strong For",
    },
    level_aliases={
        "high": "High",
        "moderate": "Moderate",
        "low": "Low",
        "very low": "Very Low",
        "high-certainty evidence": "High",
        "moderate-certainty evidence": "Moderate",
        "low-certainty evidence": "Low",
        "very low-certainty evidence": "Very Low",
        "high certainty": "High",
        "moderate certainty": "Moderate",
        "low certainty": "Low",
        "very low certainty": "Very Low",
        # Truncation artifacts in GT data
        "igh-certainty evidence": "High",
        # Slash-separated levels (take the lower)
        "moderate/low-certainty evidence": "Low",
        # Confidence-style (used by some ERS guidelines)
        "high confidence in estimates of effect": "High",
        "moderate confidence in estimates of effect": "Moderate",
        "low confidence in estimates of effect": "Low",
        "very low confidence in estimates of effect": "Very Low",
    },
)

_SCHEMES = {
    "esc_ers": ESC_ERS,
    "abcd_123": ABCD_123,
    "grade": GRADE,
}


def get_scheme(name: str) -> GradingScheme:
    """Look up a grading scheme by name (case-insensitive)."""
    key = name.strip().lower()
    if key not in _SCHEMES:
        raise ValueError(f"Unknown grading scheme '{name}'. Available: {list(_SCHEMES.keys())}")
    return _SCHEMES[key]
