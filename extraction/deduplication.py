"""Cross-page deduplication of extracted recommendations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from similarity_evaluation.similarity_models import SimilarityModel


def deduplicate_recommendations(
    df: pd.DataFrame,
    similarity_threshold: float = 0.9,
    similarity_model: SimilarityModel | None = None,
) -> pd.DataFrame:
    """Remove duplicate recommendations from a DataFrame.

    Pass 1: Exact text match (case-insensitive, whitespace-normalized).
    Pass 2: Semantic similarity dedup (optional, requires similarity_model).

    When deduplicating, keeps the longer text variant.

    Args:
        df: DataFrame with columns [recommendation, class, LOE].
        similarity_threshold: Threshold for semantic similarity dedup.
        similarity_model: Optional model for semantic dedup.

    Returns:
        Deduplicated DataFrame.
    """
    if df.empty:
        return df.copy()

    # Pass 1: Exact dedup (case-insensitive, stripped)
    df = df.copy()
    df["_norm"] = df["recommendation"].str.strip().str.lower().str.replace(r"\s+", " ", regex=True)

    # Group by normalized text, keep the row with the longest original recommendation
    keep_indices = []
    for _, group in df.groupby("_norm"):
        longest_idx = group["recommendation"].str.len().idxmax()
        keep_indices.append(longest_idx)

    df = df.loc[keep_indices].drop(columns=["_norm"]).reset_index(drop=True)

    # Pass 2: Semantic dedup (optional)
    if similarity_model is not None and len(df) > 1:
        df = _semantic_dedup(df, similarity_threshold, similarity_model)

    return df


def _semantic_dedup(
    df: pd.DataFrame,
    threshold: float,
    similarity_model: SimilarityModel,
) -> pd.DataFrame:
    """Remove semantically similar recommendations, keeping the longer one."""
    texts = df["recommendation"].tolist()
    n = len(texts)
    to_remove = set()

    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            sim = similarity_model.compute_similarity(texts[i], texts[j])
            if sim >= threshold:
                # Keep the longer recommendation
                if len(texts[i]) >= len(texts[j]):
                    to_remove.add(j)
                else:
                    to_remove.add(i)
                    break  # i is removed, stop comparing

    keep_indices = [i for i in range(n) if i not in to_remove]
    return df.iloc[keep_indices].reset_index(drop=True)
