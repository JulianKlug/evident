import pytest
import pandas as pd

from evaluation.evaluate import evaluate_extraction
from evaluation.grading import get_scheme


class TestEvaluateExtraction:
    def test_perfect_self_evaluation(self, sample_gt_df, fake_similarity_model):
        """GT evaluated against itself should give perfect scores."""
        result = evaluate_extraction(
            sample_gt_df, sample_gt_df,
            similarity_model=fake_similarity_model,
            similarity_threshold=0.5,
        )
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)
        assert result.f1 == pytest.approx(1.0)
        assert result.grade_accuracy == pytest.approx(1.0)
        assert result.level_accuracy == pytest.approx(1.0)
        assert result.combined_accuracy == pytest.approx(1.0)
        assert result.n_tp == len(sample_gt_df)
        assert result.n_fp == 0
        assert result.n_fn == 0

    def test_partial_extraction(self, sample_extracted_df, sample_gt_df, fake_similarity_model):
        """Partial extraction should produce correct completeness metrics."""
        result = evaluate_extraction(
            sample_extracted_df, sample_gt_df,
            similarity_model=fake_similarity_model,
            similarity_threshold=0.3,
        )
        # Should have matches, FP, and FN
        assert result.n_tp > 0
        assert result.n_fp > 0
        assert result.n_fn > 0
        assert 0 < result.precision < 1
        assert 0 < result.recall < 1

    def test_empty_extraction(self, empty_df, sample_gt_df, fake_similarity_model):
        """Empty extraction → precision=0, recall=0."""
        result = evaluate_extraction(
            empty_df, sample_gt_df,
            similarity_model=fake_similarity_model,
        )
        assert result.n_tp == 0
        assert result.n_fn == len(sample_gt_df)
        assert result.precision == pytest.approx(0.0)
        assert result.recall == pytest.approx(0.0)

    def test_with_grading_scheme(self, sample_gt_df, fake_similarity_model):
        """Grading scheme normalization should work in full pipeline."""
        result = evaluate_extraction(
            sample_gt_df, sample_gt_df,
            grading_scheme=get_scheme("esc_ers"),
            similarity_model=fake_similarity_model,
            similarity_threshold=0.5,
        )
        assert result.grade_accuracy == pytest.approx(1.0)
        assert result.level_accuracy == pytest.approx(1.0)

    def test_result_has_all_fields(self, sample_extracted_df, sample_gt_df, fake_similarity_model):
        """EvaluationResult should have all expected fields populated."""
        result = evaluate_extraction(
            sample_extracted_df, sample_gt_df,
            similarity_model=fake_similarity_model,
            similarity_threshold=0.3,
        )
        assert isinstance(result.matches, list)
        assert isinstance(result.false_positives, pd.DataFrame)
        assert isinstance(result.false_negatives, pd.DataFrame)
        assert isinstance(result.grade_confusion, pd.DataFrame)
        assert isinstance(result.level_confusion, pd.DataFrame)
        assert isinstance(result.per_class_grade, dict)
        assert isinstance(result.per_class_level, dict)
