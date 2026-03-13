"""Post-extraction classification filter: binary YES/NO filter on each extracted candidate."""

from __future__ import annotations

import pandas as pd

from extraction.llm_client import OllamaClient
from extraction.prompts import _SCHEME_TERMINOLOGY
from evaluation.grading import GradingScheme


_CLASSIFICATION_PROMPT = (
    "You are a clinical guideline expert.\n"
    "\n"
    "A clinical recommendation is an actionable statement that DIRECTS clinical practice "
    "and is EXPLICITLY graded with a {grade_label} and a {level_label}.\n"
    "\n"
    'Text: "{recommendation_text}"\n'
    'Assigned {grade_label}: "{grade}"\n'
    'Assigned {level_label}: "{level}"\n'
    "\n"
    "Is this a genuine clinical recommendation, or is it something else "
    "(background, evidence summary, methodology, commentary)?\n"
    "\n"
    "Answer YES or NO only.\n"
    "Answer:"
)


def classify_recommendations(
    df: pd.DataFrame,
    client: OllamaClient,
    scheme: GradingScheme,
) -> pd.DataFrame:
    """Filter DataFrame to only rows classified as genuine recommendations.

    Args:
        df: DataFrame with columns [recommendation, class, LOE].
        client: OllamaClient instance (e.g. qwen3:8b) for classification.
        scheme: GradingScheme to determine terminology.

    Returns:
        Filtered DataFrame containing only rows classified as recommendations.
    """
    if df.empty:
        return df

    terms = _SCHEME_TERMINOLOGY.get(scheme.name, {
        "grade_label": "grade",
        "level_label": "level of evidence",
    })
    grade_label = terms["grade_label"]
    level_label = terms["level_label"]

    keep_mask = []
    for _, row in df.iterrows():
        prompt = _CLASSIFICATION_PROMPT.format(
            grade_label=grade_label,
            level_label=level_label,
            recommendation_text=row.get("recommendation", ""),
            grade=row.get("class", ""),
            level=row.get("LOE", ""),
        )
        response = client.generate(prompt)
        answer = response.raw_text.strip().upper()
        keep_mask.append(answer.startswith("YES"))

    n_kept = sum(keep_mask)
    n_removed = len(keep_mask) - n_kept
    print(f"  [Filter] Kept {n_kept}/{len(df)} recommendations "
          f"(removed {n_removed} non-recs)", flush=True)

    return df[keep_mask].reset_index(drop=True)
