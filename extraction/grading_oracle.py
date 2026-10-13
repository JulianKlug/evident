"""Post-extraction grading oracle: re-grade recommendations using a reasoning model."""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np
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


_CONTEXT_REGRADE_PROMPT = (
    "You are a clinical guideline grading expert.\n\n"
    "Below is the relevant source text from a clinical guideline, followed by "
    "an extracted recommendation with its assigned {grade_label} and {level_label}.\n\n"
    "--- SOURCE TEXT ---\n"
    "{source_context}\n"
    "--- END SOURCE TEXT ---\n\n"
    'Recommendation: "{recommendation_text}"\n'
    'Current {grade_label}: "{grade}"\n'
    'Current {level_label}: "{level}"\n\n'
    "Valid {grade_label} values: {grade_values}\n"
    "Valid {level_label} values: {level_values}\n\n"
    "Instructions:\n"
    "1. Find this recommendation (or its closest match) in the source text.\n"
    "2. Check what {grade_label} and {level_label} are stated in the source.\n"
    "3. If the source clearly states different values, correct them. "
    "If you cannot find this recommendation or the values are ambiguous, "
    "keep the current values.\n\n"
    "Output EXACTLY two lines:\n"
    "{grade_label}: <value>\n"
    "{level_label}: <value>\n"
)


def regrade_with_context(
    df: pd.DataFrame,
    scheme: GradingScheme,
    page_texts: List[str],
    similarity_model,
    model: str = "deepseek-r1:32b",
    top_k: int = 2,
    page_embeddings: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Re-grade recommendations using a reasoning model with source context.

    For each recommendation, retrieves the most relevant source pages using
    BioLORD similarity, then asks the oracle to verify/correct grades with
    that context.

    Args:
        df: DataFrame with columns [recommendation, class, LOE].
        scheme: GradingScheme for terminology and valid values.
        page_texts: List of page text strings from the source PDF.
        similarity_model: Model with encode_batch() for BioLORD embeddings.
        model: Ollama model to use as grading oracle.
        top_k: Number of most similar pages to include as context.
        page_embeddings: Pre-computed page embeddings (avoids recomputing).

    Returns:
        DataFrame with potentially updated class and LOE columns.
    """
    if df.empty or not page_texts:
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

    # Encode all pages with BioLORD (once)
    if page_embeddings is None:
        page_embeddings = similarity_model.encode_batch(page_texts)

    client = OllamaClient(model=model)
    df = df.copy()

    n_grade_changed = 0
    n_level_changed = 0

    for idx, row in df.iterrows():
        rec_text = row.get("recommendation", "")
        if not rec_text:
            continue

        # Encode recommendation and find top-k most similar pages
        rec_embedding = similarity_model.encode_batch([rec_text])  # (1, D)
        similarities = (rec_embedding @ page_embeddings.T).flatten()  # (N,)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Assemble context from top-k pages
        context_parts = []
        for page_idx in sorted(top_indices):
            context_parts.append(f"[Page {page_idx + 1}]\n{page_texts[page_idx]}")
        source_context = "\n\n".join(context_parts)

        prompt = _CONTEXT_REGRADE_PROMPT.format(
            grade_label=grade_label,
            level_label=level_label,
            recommendation_text=rec_text,
            grade=row.get("class", ""),
            level=row.get("LOE", ""),
            grade_values=grade_values,
            level_values=level_values,
            source_context=source_context,
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

    print(f"  [Context Oracle] Re-graded {len(df)} recs: "
          f"{n_grade_changed} grades changed, {n_level_changed} levels changed",
          flush=True)

    return df
