"""Post-extraction grading oracle: re-grade recommendations using a reasoning model."""

from __future__ import annotations

import re

import pandas as pd

from extraction.llm_client import OllamaClient
from extraction.prompts import _SCHEME_TERMINOLOGY
from evaluation.grading import GradingScheme


_REGRADE_PROMPT = (
    "You are a clinical guideline grading expert.\n\n"
    "Verify the assigned {grade_label} and {level_label} for this recommendation.\n\n"
    'Recommendation: "{recommendation_text}"\n'
    'Current {grade_label}: "{grade}"\n'
    'Current {level_label}: "{level}"\n\n'
    "Valid {grade_label} values: {grade_values}\n"
    "Valid {level_label} values: {level_values}\n\n"
    "Output EXACTLY two lines:\n"
    "{grade_label}: <value>\n"
    "{level_label}: <value>\n"
)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from deepseek-r1 output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_regrade_response(
    text: str,
    grade_label: str,
    level_label: str,
    valid_grades: set,
    valid_levels: set,
):
    """Parse grade and level from oracle response.

    Returns:
        (grade, level) tuple, or (None, None) if parsing fails.
    """
    text = _strip_thinking(text)

    grade = None
    level = None

    for line in text.split("\n"):
        line = line.strip()
        # Match "label: value" pattern (case-insensitive on label)
        m = re.match(r"^(.+?):\s*(.+)$", line)
        if not m:
            continue
        label = m.group(1).strip().lower()
        value = m.group(2).strip()

        if label == grade_label.lower():
            if value.lower() in {v.lower() for v in valid_grades}:
                # Use the canonical casing from valid_grades
                for v in valid_grades:
                    if v.lower() == value.lower():
                        grade = v
                        break
        elif label == level_label.lower():
            if value.lower() in {v.lower() for v in valid_levels}:
                for v in valid_levels:
                    if v.lower() == value.lower():
                        level = v
                        break

    return grade, level


def regrade_recommendations(
    df: pd.DataFrame,
    scheme: GradingScheme,
    model: str = "deepseek-r1:32b",
) -> pd.DataFrame:
    """Re-grade recommendations using a reasoning model.

    For each row, sends the recommendation text and current grade/level to
    the oracle model, which verifies or corrects the values. Only updates
    values when the oracle returns valid, parseable results.

    Args:
        df: DataFrame with columns [recommendation, class, LOE].
        scheme: GradingScheme for terminology and valid values.
        model: Ollama model to use as grading oracle.

    Returns:
        DataFrame with potentially updated class and LOE columns.
    """
    if df.empty:
        return df

    terms = _SCHEME_TERMINOLOGY.get(scheme.name, {
        "grade_label": "grade",
        "level_label": "level of evidence",
        "grade_values": ", ".join(scheme.grades),
        "level_values": ", ".join(scheme.levels),
    })
    grade_label = terms["grade_label"]
    level_label = terms["level_label"]
    grade_values = terms.get("grade_values", ", ".join(scheme.grades))
    level_values = terms.get("level_values", ", ".join(scheme.levels))

    valid_grades = set(scheme.grades)
    valid_levels = set(scheme.levels)

    client = OllamaClient(model=model)
    df = df.copy()

    n_grade_changed = 0
    n_level_changed = 0

    for idx, row in df.iterrows():
        prompt = _REGRADE_PROMPT.format(
            grade_label=grade_label,
            level_label=level_label,
            recommendation_text=row.get("recommendation", ""),
            grade=row.get("class", ""),
            level=row.get("LOE", ""),
            grade_values=grade_values,
            level_values=level_values,
        )
        response = client.generate(prompt)
        new_grade, new_level = _parse_regrade_response(
            response.raw_text, grade_label, level_label,
            valid_grades, valid_levels,
        )

        if new_grade is not None and new_grade != row.get("class", ""):
            df.at[idx, "class"] = new_grade
            n_grade_changed += 1
        if new_level is not None and new_level != row.get("LOE", ""):
            df.at[idx, "LOE"] = new_level
            n_level_changed += 1

    print(f"  [Oracle] Re-graded {len(df)} recs: "
          f"{n_grade_changed} grades changed, {n_level_changed} levels changed",
          flush=True)

    return df
