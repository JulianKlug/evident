"""Post-extraction normalization of grades and levels."""

from __future__ import annotations

import pandas as pd

from evaluation.grading import GradingScheme
from extraction.response_parser import CATEGORY_GRADED


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

    # Ungraded statements keep their sentinel tokens (e.g. BEST_PRACTICE | NA)
    graded = result["category"] == CATEGORY_GRADED if "category" in result.columns else slice(None)
    result.loc[graded, "class"] = result.loc[graded, "class"].apply(
        lambda x: scheme.normalize_grade(str(x)) or str(x)
    )
    result.loc[graded, "LOE"] = result.loc[graded, "LOE"].apply(
        lambda x: scheme.normalize_level(str(x)) or str(x)
    )
    return result
