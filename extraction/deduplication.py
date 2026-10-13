"""Cross-page deduplication of extracted recommendations."""

from __future__ import annotations

import re
import unicodedata
from enum import Enum
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from similarity_evaluation.similarity_models import SimilarityModel


class DedupKey(str, Enum):
    """How pass 1 decides two rows are the same recommendation."""
    WHITESPACE = "whitespace"      # lowercase, collapsed whitespace (legacy)
    ALPHANUMERIC = "alphanumeric"  # trailing grade stripped from the text, then letters and digits only


# "(strong recommendation; high-certainty evidence)." at the end of a recommendation
_GRADE_TAIL = re.compile(r"\((?:[^()]*?\b(?:recommendation|evidence)\b[^()]*)\)\s*\.?\s*$", re.IGNORECASE)
_NON_ALPHANUMERIC = re.compile(r"[^0-9a-z]")


def deduplicate_recommendations(
    df: pd.DataFrame,
    similarity_threshold: float = 0.9,
    similarity_model: SimilarityModel | None = None,
    key: DedupKey = DedupKey.WHITESPACE,
) -> pd.DataFrame:
    """Remove duplicate recommendations from a DataFrame.

    Pass 1: Exact text match on `key` (default: case-insensitive, whitespace-normalized).
    Pass 2: Semantic similarity dedup (optional, requires similarity_model).

    DedupKey.ALPHANUMERIC merges copies that differ only in PDF text noise, e.g.
    "…lifestyle modification" and "…life- style modiﬁcation (strong recommendation; high-certainty evidence)."

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
    if key == DedupKey.ALPHANUMERIC:
        # The grade has its own columns; in the text it only hides duplicates and lowers GT similarity
        df["recommendation"] = df["recommendation"].map(_strip_grade_tail)
        df["_norm"] = df["recommendation"].map(_alphanumeric_key)
    else:
        df["_norm"] = df["recommendation"].str.strip().str.lower().str.replace(r"\s+", " ", regex=True)

    # Group by normalized text, keep the row with the longest original recommendation
    has_page = "page" in df.columns
    keep_indices = []
    for _, group in df.groupby("_norm"):
        longest_idx = group["recommendation"].str.len().idxmax()
        keep_indices.append(longest_idx)
        # Provenance: the kept row points to the earliest page it appeared on
        if has_page:
            df.loc[longest_idx, "page"] = group["page"].min()

    df = df.loc[keep_indices].drop(columns=["_norm"]).reset_index(drop=True)

    # Pass 2: Semantic dedup (optional)
    if similarity_model is not None and len(df) > 1:
        df = _semantic_dedup(df, similarity_threshold, similarity_model)

    return df


def _strip_grade_tail(text: str) -> str:
    return _GRADE_TAIL.sub("", str(text).strip()).strip()


def _alphanumeric_key(text: str) -> str:
    """NFKC (ligatures → letters), lowercase, letters and digits only: "modi ﬁca- tion" → "modification"."""
    return _NON_ALPHANUMERIC.sub("", unicodedata.normalize("NFKC", text).lower())


def _semantic_dedup(
    df: pd.DataFrame,
    threshold: float,
    similarity_model: SimilarityModel,
) -> pd.DataFrame:
    """Remove semantically similar recommendations, keeping the longer one."""
    texts = df["recommendation"].tolist()
    pages = df["page"].tolist() if "page" in df.columns else None
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
                # Kept row inherits the earliest page of the pair
                if pages is not None:
                    pages[i] = pages[j] = min(pages[i], pages[j])
                # Keep the longer recommendation
                if len(texts[i]) >= len(texts[j]):
                    to_remove.add(j)
                else:
                    to_remove.add(i)
                    break  # i is removed, stop comparing

    keep_indices = [i for i in range(n) if i not in to_remove]
    if pages is not None:
        df = df.assign(page=pages)
    return df.iloc[keep_indices].reset_index(drop=True)
