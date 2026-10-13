"""Tests for extraction.postprocessing."""

import pandas as pd

from evaluation.grading import GRADE
from extraction.postprocessing import normalize_extracted_grades


def test_ungraded_rows_untouched():
    df = pd.DataFrame({
        "recommendation": ["a", "b"],
        "class": ["strong recommendation", "NO_RECOMMENDATION"],
        "LOE": ["high-certainty evidence", "NA"],
        "category": ["graded", "no_recommendation"],
    })

    out = normalize_extracted_grades(df, GRADE)

    assert out["class"].tolist() == ["Strong For", "NO_RECOMMENDATION"]
    assert out["LOE"].tolist() == ["High", "NA"]


def test_frame_without_category_normalizes_all_rows():
    df = pd.DataFrame({"recommendation": ["a"], "class": ["strong recommendation"], "LOE": ["low"]})
    assert normalize_extracted_grades(df, GRADE)["class"].tolist() == ["Strong For"]
