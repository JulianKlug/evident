from __future__ import annotations

import argparse
import os

import pandas as pd

from evaluation.grading import GradingScheme, get_scheme
from evaluation.matching import match_recommendations, MatchResult
from evaluation.metrics import compute_metrics, EvaluationResult
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from similarity_evaluation.similarity_models import SimilarityModel


def _read_dataframe(path: str) -> pd.DataFrame:
    """Read a CSV or Excel file into a DataFrame based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xls", ".xlsx", ".xlsm", ".xlsb"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def evaluate_extraction(
    extracted_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    grading_scheme: GradingScheme | None = None,
    similarity_model: SimilarityModel | None = None,
    similarity_threshold: float = 0.95,
) -> EvaluationResult:
    """
    Full evaluation pipeline:
    1. Match recommendations (matching.py)
    2. Compute metrics (metrics.py)
    3. Return structured result
    """
    match_result = match_recommendations(
        extracted_df, gt_df,
        similarity_model=similarity_model,
        similarity_threshold=similarity_threshold,
    )
    return compute_metrics(match_result, grading_scheme=grading_scheme)


def print_report(result: EvaluationResult) -> None:
    """Print human-readable evaluation report to stdout."""
    print("=" * 60)
    print("EXTRACTION EVALUATION REPORT")
    print("=" * 60)

    print(f"\n--- Completeness ---")
    print(f"  True Positives:  {result.n_tp}")
    print(f"  False Positives: {result.n_fp}")
    print(f"  False Negatives: {result.n_fn}")
    print(f"  Precision:       {result.precision:.3f}")
    print(f"  Recall:          {result.recall:.3f}")
    print(f"  F1 Score:        {result.f1:.3f}")

    print(f"\n--- Match Quality ---")
    print(f"  Mean Similarity: {result.mean_similarity:.3f}")
    print(f"  Min Similarity:  {result.min_similarity:.3f}")
    print(f"  Max Similarity:  {result.max_similarity:.3f}")

    if result.matches:
        weakest = sorted(result.matches, key=lambda m: m.similarity_score)[:3]
        print(f"\n--- Weakest Matches (bottom {len(weakest)}) ---")
        for m in weakest:
            print(f"  [{m.similarity_score:.3f}] {m.extracted_text[:80]}")
            print(f"       vs {m.gt_text[:80]}")

    print(f"\n--- Accuracy (matched only, n={result.n_tp}) ---")
    print(f"  Grade Accuracy:    {result.grade_accuracy:.3f}")
    print(f"  Level Accuracy:    {result.level_accuracy:.3f}")
    print(f"  Combined Accuracy: {result.combined_accuracy:.3f}")

    if result.per_class_grade:
        print(f"\n--- Per-Class Grade Accuracy ---")
        for cls, acc in sorted(result.per_class_grade.items()):
            print(f"  {cls}: {acc:.3f}")

    if result.per_class_level:
        print(f"\n--- Per-Class Level Accuracy ---")
        for cls, acc in sorted(result.per_class_level.items()):
            print(f"  {cls}: {acc:.3f}")

    if not result.grade_confusion.empty:
        print(f"\n--- Grade Confusion Matrix (rows=actual, cols=predicted) ---")
        print(result.grade_confusion.to_string())

    if not result.level_confusion.empty:
        print(f"\n--- Level Confusion Matrix (rows=actual, cols=predicted) ---")
        print(result.level_confusion.to_string())

    if len(result.false_negatives) > 0:
        print(f"\n--- Missing Recommendations ({len(result.false_negatives)}) ---")
        for _, row in result.false_negatives.iterrows():
            print(f"  - [{row.get('class', '?')}/{row.get('LOE', '?')}] {row['recommendation'][:100]}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate extraction against ground truth")
    parser.add_argument("-p", "--extracted", type=str, required=True, help="Path to extracted recommendations (CSV or Excel)")
    parser.add_argument("-g", "--ground-truth", type=str, required=True, help="Path to ground truth (CSV or Excel)")
    parser.add_argument("-s", "--scheme", type=str, default=None, help="Grading scheme name (esc_ers, abcd_123, grade)")
    parser.add_argument("-t", "--threshold", type=float, default=0.95, help="Similarity threshold (default 0.95)")
    parser.add_argument("-m", "--model", type=str, default=None, help="Similarity model name (default: FremyCompany/BioLORD-2023)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output directory for results CSVs")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    extracted_df = _read_dataframe(args.extracted)
    gt_df = _read_dataframe(args.ground_truth)

    grading_scheme = get_scheme(args.scheme) if args.scheme else None

    similarity_model = None
    if args.model:
        from similarity_evaluation.similarity_models import get_similarity_model
        similarity_model = get_similarity_model(args.model)

    result = evaluate_extraction(
        extracted_df, gt_df,
        grading_scheme=grading_scheme,
        similarity_model=similarity_model,
        similarity_threshold=args.threshold,
    )

    print_report(result)

    if args.output:
        os.makedirs(args.output, exist_ok=True)

        # Save matches
        matches_data = [
            {
                "extracted_text": m.extracted_text,
                "extracted_grade": m.extracted_grade,
                "extracted_level": m.extracted_level,
                "gt_text": m.gt_text,
                "gt_grade": m.gt_grade,
                "gt_level": m.gt_level,
                "similarity_score": m.similarity_score,
            }
            for m in result.matches
        ]
        pd.DataFrame(matches_data).to_csv(os.path.join(args.output, "matches.csv"), index=False)
        result.false_positives.to_csv(os.path.join(args.output, "false_positives.csv"), index=False)
        result.false_negatives.to_csv(os.path.join(args.output, "false_negatives.csv"), index=False)

        if args.verbose:
            print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()
