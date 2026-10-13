"""
Tests that evaluate matching quality against the annotated matching evaluation dataset.

The dataset at MATCHING_EVAL_PATH contains recommendation pairs with a ground truth
'match' column (1 = same recommendation, 0 = different recommendation).
Non-matching pairs are semantically distinct (e.g. opposite clinical meaning) despite
high lexical overlap, so this tests whether the similarity model captures meaning.
"""
import os

import pytest
import pandas as pd
import numpy as np

from evaluation.matching import build_similarity_matrix
from tests.conftest import MATCHING_EVAL_PATH

needs_dataset = pytest.mark.skipif(
    not os.path.exists(MATCHING_EVAL_PATH),
    reason=f"Matching evaluation dataset not found at {MATCHING_EVAL_PATH}",
)

try:
    import similarity_evaluation.similarity_models  # noqa: F401
    _has_similarity_models = True
except ImportError:
    _has_similarity_models = False

needs_similarity_models = pytest.mark.skipif(
    not _has_similarity_models,
    reason="similarity_evaluation dependencies (spacy, sentence_transformers) not installed",
)


@needs_dataset
class TestMatchingEvaluation:
    """Evaluate pairwise similarity scores against ground truth match labels."""

    def test_dataset_loads(self, matching_eval_df):
        assert len(matching_eval_df) > 0
        assert set(matching_eval_df.columns) >= {"recommendation_a", "recommendation_b", "match"}
        assert matching_eval_df["match"].isin([0, 1]).all()

    def test_fake_model_cannot_separate_pairs(self, matching_eval_df, fake_similarity_model):
        """
        Word-overlap model CANNOT separate these pairs because non-matching pairs
        differ by only 1-2 words with opposite meaning (e.g. 'against' removed,
        'crystalloids' vs 'albumin' swapped). This demonstrates why a semantic
        model is needed.
        """
        scores = []
        for _, row in matching_eval_df.iterrows():
            score = fake_similarity_model.compute_similarity(
                row["recommendation_a"], row["recommendation_b"]
            )
            scores.append(score)

        matching_eval_df = matching_eval_df.copy()
        matching_eval_df["score"] = scores

        match_scores = matching_eval_df.loc[matching_eval_df["match"] == 1, "score"]
        non_match_scores = matching_eval_df.loc[matching_eval_df["match"] == 0, "score"]

        # Both groups have very high word overlap — fake model can't tell them apart
        assert match_scores.mean() > 0.8
        assert non_match_scores.mean() > 0.8

    @pytest.mark.slow
    @needs_similarity_models
    def test_sentence_transformer_classification(self, matching_eval_df):
        """
        Test that PubMedBERT sentence transformer can correctly classify
        matching vs non-matching recommendation pairs using a threshold.
        """
        from similarity_evaluation.similarity_models import SentenceTransformerSimilarityModel

        model = SentenceTransformerSimilarityModel("neuml/pubmedbert-base-embeddings")

        scores = []
        for _, row in matching_eval_df.iterrows():
            score = model.compute_similarity(
                row["recommendation_a"], row["recommendation_b"]
            )
            scores.append(score)

        matching_eval_df = matching_eval_df.copy()
        matching_eval_df["score"] = scores

        match_scores = matching_eval_df.loc[matching_eval_df["match"] == 1, "score"]
        non_match_scores = matching_eval_df.loc[matching_eval_df["match"] == 0, "score"]

        # Matching pairs should have strictly higher mean similarity
        assert match_scores.mean() > non_match_scores.mean(), (
            f"Mean match score ({match_scores.mean():.3f}) should exceed "
            f"mean non-match score ({non_match_scores.mean():.3f})"
        )

        # Find the best threshold (maximize accuracy)
        all_thresholds = np.arange(0.5, 1.0, 0.01)
        labels = matching_eval_df["match"].values
        score_arr = np.array(scores)

        best_acc = 0.0
        best_threshold = 0.0
        for t in all_thresholds:
            predicted = (score_arr >= t).astype(int)
            acc = (predicted == labels).mean()
            if acc > best_acc:
                best_acc = acc
                best_threshold = t

        # The model should achieve at least 80% accuracy at the best threshold
        assert best_acc >= 0.8, (
            f"Best accuracy {best_acc:.3f} at threshold {best_threshold:.2f} is below 0.8"
        )

        # At the default threshold of 0.6, check performance
        predicted_default = (score_arr >= 0.6).astype(int)
        acc_default = (predicted_default == labels).mean()

        # Print summary for visibility when run with -v
        print(f"\n--- Matching Evaluation Summary ---")
        print(f"  Pairs: {len(matching_eval_df)} ({labels.sum()} match, {(1-labels).sum()} non-match)")
        print(f"  Match scores:     mean={match_scores.mean():.3f}, min={match_scores.min():.3f}, max={match_scores.max():.3f}")
        print(f"  Non-match scores: mean={non_match_scores.mean():.3f}, min={non_match_scores.min():.3f}, max={non_match_scores.max():.3f}")
        print(f"  Best threshold:   {best_threshold:.2f} (accuracy={best_acc:.3f})")
        print(f"  Default (0.6):    accuracy={acc_default:.3f}")

    @pytest.mark.slow
    @needs_similarity_models
    def test_build_similarity_matrix_with_real_model(self, matching_eval_df):
        """Test that build_similarity_matrix works with real data and model."""
        from similarity_evaluation.similarity_models import SentenceTransformerSimilarityModel

        model = SentenceTransformerSimilarityModel("neuml/pubmedbert-base-embeddings")

        texts_a = matching_eval_df["recommendation_a"].tolist()
        texts_b = matching_eval_df["recommendation_b"].tolist()

        matrix = build_similarity_matrix(texts_a, texts_b, model)
        assert matrix.shape == (len(texts_a), len(texts_b))

        # Diagonal should correspond to the paired comparisons
        diagonal_scores = np.diag(matrix)
        labels = matching_eval_df["match"].values

        # Matching pairs (on diagonal) should have higher similarity than non-matching
        match_diag = diagonal_scores[labels == 1]
        non_match_diag = diagonal_scores[labels == 0]
        assert match_diag.mean() > non_match_diag.mean()
