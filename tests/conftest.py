import os

import pytest
import pandas as pd

from evaluation.matching import MatchResult, RecommendationMatch
from evaluation.grading import GradingScheme

MATCHING_EVAL_PATH = "/mnt/data1/klug/datasets/evidence_extraction/matching_evaluation.xlsx"


class FakeSimilarityModel:
    """A mock similarity model that uses simple word overlap for fast tests."""

    def __init__(self):
        self.name = "fake"

    def compute_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)


@pytest.fixture
def fake_similarity_model():
    return FakeSimilarityModel()


@pytest.fixture
def sample_gt_df():
    """Ground truth with 6 recommendations."""
    return pd.DataFrame({
        "recommendation": [
            "Patients with heart failure should receive beta-blockers",
            "ACE inhibitors are recommended for hypertension",
            "Statins should be prescribed for high cholesterol",
            "Anticoagulation therapy is recommended for atrial fibrillation",
            "Diuretics are first-line for fluid overload",
            "Cardiac rehabilitation is recommended after myocardial infarction",
        ],
        "class": ["I", "I", "IIa", "I", "IIb", "I"],
        "LOE": ["A", "A", "B", "A", "C", "B"],
    })


@pytest.fixture
def sample_extracted_df():
    """
    Extracted recommendations: 5 total.
    - 3 match GT well (2 correct grades, 1 wrong grade)
    - 1 partial match (wrong level)
    - 1 has no match in GT (false positive)
    Missing from GT: "Diuretics..." and "Cardiac rehabilitation..." (false negatives)
    """
    return pd.DataFrame({
        "recommendation": [
            "Patients with heart failure should receive beta-blockers",       # exact match, correct
            "ACE inhibitors are recommended for hypertension",                # exact match, correct
            "Statins should be prescribed for high cholesterol",              # exact match, wrong grade
            "Anticoagulation therapy is recommended for atrial fibrillation", # exact match, wrong level
            "Aspirin should be given to all patients daily",                  # no match (FP)
        ],
        "class": ["I", "I", "I", "I", "I"],     # 3rd is wrong (should be IIa)
        "LOE": ["A", "A", "B", "B", "A"],        # 4th is wrong (should be A)
    })


@pytest.fixture
def perfect_extracted_df(sample_gt_df):
    """Extracted that perfectly matches GT."""
    return sample_gt_df.copy()


@pytest.fixture
def empty_df():
    return pd.DataFrame(columns=["recommendation", "class", "LOE"])


@pytest.fixture
def simple_scheme():
    return GradingScheme(
        name="test",
        grades=["I", "IIa", "IIb", "III"],
        levels=["A", "B", "C"],
        grade_aliases={"1": "I", "2a": "IIa"},
        level_aliases={},
    )


@pytest.fixture
def matching_eval_df():
    """Load the matching evaluation dataset (recommendation pairs with ground truth labels)."""
    return pd.read_excel(MATCHING_EVAL_PATH)
