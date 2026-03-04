"""Tests for extraction.deduplication."""

import pandas as pd
import pytest

from extraction.deduplication import deduplicate_recommendations


def _make_df(recs, classes=None, loes=None):
    n = len(recs)
    return pd.DataFrame({
        "recommendation": recs,
        "class": classes or ["A"] * n,
        "LOE": loes or ["1"] * n,
    })


class TestExactDedup:
    def test_removes_exact_duplicates(self):
        df = _make_df(["Use drug X", "Use drug X", "Use drug Y"])
        result = deduplicate_recommendations(df)
        assert len(result) == 2

    def test_case_insensitive(self):
        df = _make_df(["Use drug X", "use drug x", "Use Drug Y"])
        result = deduplicate_recommendations(df)
        assert len(result) == 2

    def test_whitespace_normalized(self):
        df = _make_df(["Use drug X", "  Use  drug  X  ", "Use drug Y"])
        result = deduplicate_recommendations(df)
        assert len(result) == 2

    def test_keeps_longer_variant(self):
        df = _make_df(["Use drug X for condition", "Use drug X for condition Y"])
        # These are different after normalization, so both should be kept
        result = deduplicate_recommendations(df)
        assert len(result) == 2

    def test_exact_match_keeps_longer(self):
        # Same content after case normalization, different length
        df = _make_df(["Use Drug X", "use drug x"])
        result = deduplicate_recommendations(df)
        assert len(result) == 1
        assert result.iloc[0]["recommendation"] == "Use Drug X"  # Longer variant

    def test_no_false_dedup(self):
        df = _make_df(["Use drug X for condition A", "Use drug X for condition B"])
        result = deduplicate_recommendations(df)
        assert len(result) == 2

    def test_empty_df(self):
        df = pd.DataFrame(columns=["recommendation", "class", "LOE"])
        result = deduplicate_recommendations(df)
        assert result.empty


class TestSemanticDedup:
    def test_with_mock_model(self):
        """Semantic dedup with a model that always returns high similarity."""
        class AlwaysSimilar:
            def compute_similarity(self, t1, t2):
                return 0.95

        df = _make_df(["Use drug X for HF", "Prescribe drug X for heart failure"])
        result = deduplicate_recommendations(df, similarity_threshold=0.9, similarity_model=AlwaysSimilar())
        assert len(result) == 1

    def test_no_semantic_dedup_without_model(self):
        df = _make_df(["Use drug X for HF", "Prescribe drug X for heart failure"])
        result = deduplicate_recommendations(df, similarity_threshold=0.9, similarity_model=None)
        # Without a model, only exact dedup applies; these are different
        assert len(result) == 2
