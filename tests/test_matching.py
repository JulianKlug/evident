import pandas as pd
import pytest

from evaluation.matching import match_recommendations, build_similarity_matrix


class TestBuildSimilarityMatrix:
    def test_shape(self, fake_similarity_model):
        extracted = ["hello world", "foo bar"]
        gt = ["hello world", "baz qux", "foo bar"]
        matrix = build_similarity_matrix(extracted, gt, fake_similarity_model)
        assert matrix.shape == (2, 3)

    def test_identical_texts_high_similarity(self, fake_similarity_model):
        texts = ["the quick brown fox"]
        matrix = build_similarity_matrix(texts, texts, fake_similarity_model)
        assert matrix[0, 0] == pytest.approx(1.0)


class TestMatchRecommendations:
    def test_perfect_match(self, sample_gt_df, perfect_extracted_df, fake_similarity_model):
        """Identical DataFrames → all matched, no FP/FN."""
        result = match_recommendations(
            perfect_extracted_df, sample_gt_df,
            similarity_model=fake_similarity_model,
            similarity_threshold=0.5,
        )
        assert len(result.matches) == len(sample_gt_df)
        assert len(result.false_positives) == 0
        assert len(result.false_negatives) == 0

    def test_partial_match(self, sample_gt_df, sample_extracted_df, fake_similarity_model):
        """Some matching, some FP, some FN."""
        result = match_recommendations(
            sample_extracted_df, sample_gt_df,
            similarity_model=fake_similarity_model,
            similarity_threshold=0.3,
        )
        # 4 exact text matches should be found, 1 FP (aspirin), 2 FN
        assert len(result.matches) == 4
        assert len(result.false_positives) == 1
        assert len(result.false_negatives) == 2

    def test_no_duplicate_matching(self, fake_similarity_model):
        """Two very similar extractions should not both match the same GT."""
        extracted_df = pd.DataFrame({
            "recommendation": [
                "Patients with heart failure should receive beta-blockers",
                "Patients with heart failure should receive beta-blockers treatment",
            ],
            "class": ["I", "I"],
            "LOE": ["A", "A"],
        })
        gt_df = pd.DataFrame({
            "recommendation": [
                "Patients with heart failure should receive beta-blockers",
                "Statins should be prescribed for high cholesterol",
            ],
            "class": ["I", "IIa"],
            "LOE": ["A", "B"],
        })
        result = match_recommendations(
            extracted_df, gt_df,
            similarity_model=fake_similarity_model,
            similarity_threshold=0.3,
        )
        # Each GT recommendation can only be matched once
        matched_gt_texts = [m.gt_text for m in result.matches]
        assert len(matched_gt_texts) == len(set(matched_gt_texts))

    def test_threshold_filtering(self, fake_similarity_model):
        """High threshold filters out weak matches."""
        extracted_df = pd.DataFrame({
            "recommendation": ["completely unrelated text about weather"],
            "class": ["I"],
            "LOE": ["A"],
        })
        gt_df = pd.DataFrame({
            "recommendation": ["Patients with heart failure should receive beta-blockers"],
            "class": ["I"],
            "LOE": ["A"],
        })
        result = match_recommendations(
            extracted_df, gt_df,
            similarity_model=fake_similarity_model,
            similarity_threshold=0.9,
        )
        assert len(result.matches) == 0
        assert len(result.false_positives) == 1
        assert len(result.false_negatives) == 1

    def test_empty_extracted(self, sample_gt_df, empty_df, fake_similarity_model):
        result = match_recommendations(
            empty_df, sample_gt_df,
            similarity_model=fake_similarity_model,
        )
        assert len(result.matches) == 0
        assert len(result.false_positives) == 0
        assert len(result.false_negatives) == len(sample_gt_df)

    def test_empty_gt(self, sample_extracted_df, empty_df, fake_similarity_model):
        result = match_recommendations(
            sample_extracted_df, empty_df,
            similarity_model=fake_similarity_model,
        )
        assert len(result.matches) == 0
        assert len(result.false_positives) == len(sample_extracted_df)
        assert len(result.false_negatives) == 0

    def test_both_empty(self, empty_df, fake_similarity_model):
        result = match_recommendations(
            empty_df, empty_df.copy(),
            similarity_model=fake_similarity_model,
        )
        assert len(result.matches) == 0
        assert len(result.false_positives) == 0
        assert len(result.false_negatives) == 0


class TestInputValidation:
    def test_missing_recommendation_column(self, fake_similarity_model):
        df_bad = pd.DataFrame({"class": ["I"], "LOE": ["A"]})
        df_good = pd.DataFrame({"recommendation": ["text"], "class": ["I"], "LOE": ["A"]})
        with pytest.raises(ValueError, match="recommendation"):
            match_recommendations(df_bad, df_good, similarity_model=fake_similarity_model)

    def test_missing_class_column(self, fake_similarity_model):
        df_bad = pd.DataFrame({"recommendation": ["text"], "LOE": ["A"]})
        df_good = pd.DataFrame({"recommendation": ["text"], "class": ["I"], "LOE": ["A"]})
        with pytest.raises(ValueError, match="class"):
            match_recommendations(df_bad, df_good, similarity_model=fake_similarity_model)

    def test_missing_loe_column(self, fake_similarity_model):
        df_bad = pd.DataFrame({"recommendation": ["text"], "class": ["I"]})
        df_good = pd.DataFrame({"recommendation": ["text"], "class": ["I"], "LOE": ["A"]})
        with pytest.raises(ValueError, match="LOE"):
            match_recommendations(df_good, df_bad, similarity_model=fake_similarity_model)

    def test_valid_dataframe_passes(self, fake_similarity_model):
        df = pd.DataFrame({"recommendation": ["text"], "class": ["I"], "LOE": ["A"]})
        result = match_recommendations(df, df.copy(), similarity_model=fake_similarity_model)
        assert isinstance(result.matches, list)
