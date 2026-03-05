"""Post-extraction normalization of grades and levels."""

from __future__ import annotations

import pandas as pd

from evaluation.grading import GradingScheme


def normalize_extracted_grades(
    df: pd.DataFrame,
    scheme: GradingScheme,
) -> pd.DataFrame:
    """Normalize extracted grade and level values against the grading scheme.

    Values that can't be normalized are left as-is.

    Args:
        df: DataFrame with columns [recommendation, class, LOE].
        scheme: The grading scheme to normalize against.

    Returns:
        DataFrame with normalized class and LOE columns.
    """
    if df.empty:
        return df.copy()

    result = df.copy()
    result["class"] = result["class"].apply(
        lambda x: scheme.normalize_grade(str(x)) or str(x)
    )
    result["LOE"] = result["LOE"].apply(
        lambda x: scheme.normalize_level(str(x)) or str(x)
    )
    return result
