"""
Benchmark multiple similarity models on matching_evaluation.xlsx to find
the best model for recommendation matching.

Models span three categories:
- Bi-encoders (SentenceTransformerSimilarityModel)
- Cross-encoders (CrossEncoderSimilarityModel)
- NLI cross-encoders (NLICrossEncoderSimilarityModel)

Run with:  pytest tests/test_model_comparison.py -v -s
"""
import os

import numpy as np
import pandas as pd
import pytest

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
    reason="similarity_evaluation dependencies not installed",
)

CANDIDATE_MODELS = [
    # (model_id, model_type)
    ("neuml/pubmedbert-base-embeddings", "bi-encoder"),
    ("FremyCompany/BioLORD-2023", "bi-encoder"),
    ("ncbi/MedCPT-Query-Encoder", "bi-encoder"),
    ("cross-encoder/stsb-distilroberta-base", "cross-encoder"),
    ("cross-encoder/nli-deberta-v3-base", "nli-cross-encoder"),
]


def _make_model(model_id, model_type):
    from similarity_evaluation.similarity_models import (
        SentenceTransformerSimilarityModel,
        CrossEncoderSimilarityModel,
        NLICrossEncoderSimilarityModel,
    )
    if model_type == "bi-encoder":
        return SentenceTransformerSimilarityModel(model_id)
    elif model_type == "cross-encoder":
        return CrossEncoderSimilarityModel(model_id)
    elif model_type == "nli-cross-encoder":
        return NLICrossEncoderSimilarityModel(model_id)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _evaluate_model(model, df):
    """Compute scores and metrics for a model on the evaluation dataset."""
    scores = []
    for _, row in df.iterrows():
        score = model.compute_similarity(
            row["recommendation_a"], row["recommendation_b"]
        )
        scores.append(score)

    score_arr = np.array(scores)
    labels = df["match"].values

    match_scores = score_arr[labels == 1]
    non_match_scores = score_arr[labels == 0]

    # Sweep thresholds to find best accuracy
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_acc = 0.0
    best_threshold = 0.0
    for t in thresholds:
        predicted = (score_arr >= t).astype(int)
        acc = (predicted == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    # Accuracy at default threshold of 0.6
    predicted_default = (score_arr >= 0.6).astype(int)
    acc_default = (predicted_default == labels).mean()

    return {
        "match_mean": match_scores.mean(),
        "match_min": match_scores.min(),
        "match_max": match_scores.max(),
        "non_match_mean": non_match_scores.mean(),
        "non_match_min": non_match_scores.min(),
        "non_match_max": non_match_scores.max(),
        "separation": match_scores.mean() - non_match_scores.mean(),
        "best_accuracy": best_acc,
        "best_threshold": best_threshold,
        "acc_at_0.6": acc_default,
        "scores": scores,
    }


@needs_dataset
@needs_similarity_models
@pytest.mark.slow
class TestModelComparison:
    """Benchmark all candidate models and identify the best one."""

    @pytest.fixture(scope="class")
    def eval_df(self):
        return pd.read_excel(MATCHING_EVAL_PATH)

    @pytest.mark.parametrize(
        "model_id,model_type",
        CANDIDATE_MODELS,
        ids=[m[0].split("/")[-1] for m in CANDIDATE_MODELS],
    )
    def test_model_benchmark(self, eval_df, model_id, model_type):
        """Benchmark a single model and print its metrics."""
        model = _make_model(model_id, model_type)
        results = _evaluate_model(model, eval_df)

        print(f"\n{'='*70}")
        print(f"Model: {model_id} ({model_type})")
        print(f"  Match scores:     mean={results['match_mean']:.4f}  "
              f"min={results['match_min']:.4f}  max={results['match_max']:.4f}")
        print(f"  Non-match scores: mean={results['non_match_mean']:.4f}  "
              f"min={results['non_match_min']:.4f}  max={results['non_match_max']:.4f}")
        print(f"  Separation:       {results['separation']:.4f}")
        print(f"  Best accuracy:    {results['best_accuracy']:.3f} "
              f"(threshold={results['best_threshold']:.2f})")
        print(f"  Accuracy @0.6:    {results['acc_at_0.6']:.3f}")

    def test_comparison_summary(self, eval_df):
        """Run all models and print a comparison table. Assert best >= 90%."""
        results = {}
        for model_id, model_type in CANDIDATE_MODELS:
            model = _make_model(model_id, model_type)
            results[model_id] = _evaluate_model(model, eval_df)
            results[model_id]["type"] = model_type

        # Print comparison table
        print(f"\n{'='*90}")
        print(f"{'MODEL COMPARISON SUMMARY':^90}")
        print(f"{'='*90}")
        header = (
            f"{'Model':<45} {'Type':<18} {'BestAcc':>7} {'Thr':>5} "
            f"{'@0.6':>5} {'Sep':>7}"
        )
        print(header)
        print("-" * 90)

        best_model = None
        best_acc = 0.0
        for model_id, r in results.items():
            short_name = model_id.split("/")[-1] if "/" in model_id else model_id
            line = (
                f"{short_name:<45} {r['type']:<18} "
                f"{r['best_accuracy']:>7.3f} {r['best_threshold']:>5.2f} "
                f"{r['acc_at_0.6']:>5.3f} {r['separation']:>7.4f}"
            )
            print(line)
            if r["best_accuracy"] > best_acc:
                best_acc = r["best_accuracy"]
                best_model = model_id

        print(f"\nBest model: {best_model} (accuracy={best_acc:.3f})")
        print(f"{'='*90}")

        # The best model should beat the PubMedBERT baseline (81.8%)
        assert best_acc >= 0.9, (
            f"Best model {best_model} achieved {best_acc:.3f} accuracy, "
            f"expected >= 0.9. Consider adding more candidate models."
        )
