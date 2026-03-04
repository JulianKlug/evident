from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from evaluation.matching import MatchResult, RecommendationMatch
from evaluation.grading import GradingScheme


@dataclass
class EvaluationResult:
    # Completeness
    precision: float
    recall: float
    f1: float
    n_tp: int
    n_fp: int
    n_fn: int

    # Similarity quality
    mean_similarity: float
    min_similarity: float
    max_similarity: float

    # Accuracy (of matched recommendations only)
    grade_accuracy: float
    level_accuracy: float
    combined_accuracy: float

    # Detailed breakdowns
    grade_confusion: pd.DataFrame
    level_confusion: pd.DataFrame
    per_class_grade: dict[str, float]
    per_class_level: dict[str, float]

    # Raw data
    matches: list[RecommendationMatch]
    false_positives: pd.DataFrame
    false_negatives: pd.DataFrame


def _build_confusion_matrix(predicted: list[str], actual: list[str], labels: list[str] | None = None) -> pd.DataFrame:
    """Build a confusion matrix as a DataFrame."""
    if labels is None:
        labels = sorted(set(predicted) | set(actual))
    matrix = {label: {l2: 0 for l2 in labels} for label in labels}
    for p, a in zip(predicted, actual):
        if p in matrix and a in matrix[p]:
            matrix[a][p] += 1
    return pd.DataFrame(matrix, index=labels, columns=labels)


def _per_class_accuracy(predicted: list[str], actual: list[str]) -> dict[str, float]:
    """Compute accuracy per class (what fraction of each GT class was correctly predicted)."""
    class_correct: dict[str, int] = {}
    class_total: dict[str, int] = {}
    for p, a in zip(predicted, actual):
        class_total[a] = class_total.get(a, 0) + 1
        if p == a:
            class_correct[a] = class_correct.get(a, 0) + 1
    return {cls: class_correct.get(cls, 0) / total for cls, total in class_total.items()}


def compute_metrics(
    match_result: MatchResult,
    grading_scheme: GradingScheme | None = None,
) -> EvaluationResult:
    """
    Compute completeness and accuracy metrics from match result.
    If grading_scheme provided, normalize grades/levels before comparison.
    """
    matches = match_result.matches
    n_tp = len(matches)
    n_fp = len(match_result.false_positives)
    n_fn = len(match_result.false_negatives)

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Similarity quality stats
    if n_tp > 0:
        sim_scores = [m.similarity_score for m in matches]
        mean_similarity = sum(sim_scores) / len(sim_scores)
        min_similarity = min(sim_scores)
        max_similarity = max(sim_scores)
    else:
        mean_similarity = 0.0
        min_similarity = 0.0
        max_similarity = 0.0

    # Grade/level comparison
    if n_tp == 0:
        return EvaluationResult(
            precision=precision,
            recall=recall,
            f1=f1,
            n_tp=0,
            n_fp=n_fp,
            n_fn=n_fn,
            mean_similarity=mean_similarity,
            min_similarity=min_similarity,
            max_similarity=max_similarity,
            grade_accuracy=0.0,
            level_accuracy=0.0,
            combined_accuracy=0.0,
            grade_confusion=pd.DataFrame(),
            level_confusion=pd.DataFrame(),
            per_class_grade={},
            per_class_level={},
            matches=matches,
            false_positives=match_result.false_positives,
            false_negatives=match_result.false_negatives,
        )

    # Normalize if scheme provided
    ext_grades = []
    gt_grades = []
    ext_levels = []
    gt_levels = []

    for m in matches:
        if grading_scheme:
            eg = grading_scheme.normalize_grade(m.extracted_grade) or m.extracted_grade.strip().lower()
            gg = grading_scheme.normalize_grade(m.gt_grade) or m.gt_grade.strip().lower()
            el = grading_scheme.normalize_level(m.extracted_level) or m.extracted_level.strip().lower()
            gl = grading_scheme.normalize_level(m.gt_level) or m.gt_level.strip().lower()
        else:
            eg = m.extracted_grade.strip().lower()
            gg = m.gt_grade.strip().lower()
            el = m.extracted_level.strip().lower()
            gl = m.gt_level.strip().lower()
        ext_grades.append(eg)
        gt_grades.append(gg)
        ext_levels.append(el)
        gt_levels.append(gl)

    n_grade_correct = sum(1 for e, g in zip(ext_grades, gt_grades) if e == g)
    n_level_correct = sum(1 for e, g in zip(ext_levels, gt_levels) if e == g)
    n_both_correct = sum(
        1 for eg, gg, el, gl in zip(ext_grades, gt_grades, ext_levels, gt_levels) if eg == gg and el == gl
    )

    grade_accuracy = n_grade_correct / n_tp
    level_accuracy = n_level_correct / n_tp
    combined_accuracy = n_both_correct / n_tp

    # Confusion matrices
    grade_labels = grading_scheme.grades if grading_scheme else None
    level_labels = grading_scheme.levels if grading_scheme else None
    grade_confusion = _build_confusion_matrix(ext_grades, gt_grades, grade_labels)
    level_confusion = _build_confusion_matrix(ext_levels, gt_levels, level_labels)

    return EvaluationResult(
        precision=precision,
        recall=recall,
        f1=f1,
        n_tp=n_tp,
        n_fp=n_fp,
        n_fn=n_fn,
        mean_similarity=mean_similarity,
        min_similarity=min_similarity,
        max_similarity=max_similarity,
        grade_accuracy=grade_accuracy,
        level_accuracy=level_accuracy,
        combined_accuracy=combined_accuracy,
        grade_confusion=grade_confusion,
        level_confusion=level_confusion,
        per_class_grade=_per_class_accuracy(ext_grades, gt_grades),
        per_class_level=_per_class_accuracy(ext_levels, gt_levels),
        matches=matches,
        false_positives=match_result.false_positives,
        false_negatives=match_result.false_negatives,
    )
