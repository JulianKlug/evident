"""Post-extraction verification: filter out hallucinated recommendations.

Checks each extracted recommendation against the source text chunks that were
fed to the LLM. Recommendations with low token overlap are likely hallucinated.
"""

from __future__ import annotations

import re

import pandas as pd


def _tokenize(text: str) -> set:
    """Lowercase word tokenization."""
    return set(re.findall(r'\w+', text.lower()))


def verify_recommendations(
    df: pd.DataFrame,
    source_chunks: list,
    min_token_overlap: float = 0.5,
    fuzzy_threshold: int = 75,
) -> pd.DataFrame:
    """Filter out recommendations not grounded in source text.

    Args:
        df: DataFrame with 'recommendation' column.
        source_chunks: List of text chunks that were fed to the LLM.
        min_token_overlap: Minimum fraction of rec tokens found in source.
        fuzzy_threshold: partial_ratio threshold for borderline cases.

    Returns:
        Filtered DataFrame with ungrounded recommendations removed.
    """
    if df.empty or not source_chunks:
        return df

    # Build source token set from all chunks
    source_tokens = set()
    for chunk in source_chunks:
        source_tokens.update(_tokenize(chunk))

    keep_mask = []
    fuzz = None  # lazy import

    for _, row in df.iterrows():
        rec_text = str(row.get("recommendation", ""))
        rec_tokens = _tokenize(rec_text)

        if not rec_tokens:
            keep_mask.append(False)
            continue

        overlap = len(rec_tokens & source_tokens) / len(rec_tokens)

        if overlap >= min_token_overlap:
            keep_mask.append(True)
        elif overlap < 0.3:
            keep_mask.append(False)
        else:
            # Borderline — use fuzzy matching against individual chunks
            if fuzz is None:
                try:
                    from thefuzz import fuzz as _fuzz
                    fuzz = _fuzz
                except ImportError:
                    # No thefuzz available, fall back to token overlap only
                    keep_mask.append(False)
                    continue

            matched = any(
                fuzz.partial_ratio(rec_text.lower(), chunk.lower()) >= fuzzy_threshold
                for chunk in source_chunks
            )
            keep_mask.append(matched)

    n_total = len(df)
    filtered = df[keep_mask].reset_index(drop=True)
    n_removed = n_total - len(filtered)
    print(f"  [Verify] Kept {len(filtered)}/{n_total} recommendations "
          f"(removed {n_removed} ungrounded)", flush=True)

    return filtered
