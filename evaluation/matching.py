from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from similarity_evaluation.similarity_models import SimilarityModel


@dataclass
class RecommendationMatch:
    extracted_text: str
    extracted_grade: str
    extracted_level: str
    gt_text: str
    gt_grade: str
    gt_level: str
    similarity_score: float


@dataclass
class MatchResult:
    matches: list[RecommendationMatch]
    false_positives: pd.DataFrame  # Extracted recommendations not matched to GT
    false_negatives: pd.DataFrame  # GT recommendations not found in extraction


def build_similarity_matrix(
    extracted: list[str],
    ground_truth: list[str],
    similarity_model: SimilarityModel,
) -> np.ndarray:
    """Compute pairwise similarity matrix (n_extracted × n_gt)."""
    n_ext = len(extracted)
    n_gt = len(ground_truth)
    matrix = np.zeros((n_ext, n_gt))
    for i, ext_text in enumerate(extracted):
        for j, gt_text in enumerate(ground_truth):
            matrix[i, j] = similarity_model.compute_similarity(ext_text, gt_text)
    return matrix


def match_recommendations(
    extracted_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    similarity_model: SimilarityModel | None = None,
    similarity_threshold: float = 0.6,
) -> MatchResult:
    """
    Optimal bipartite matching between extracted and GT recommendations.

    1. Build similarity matrix using the SimilarityModel
    2. Run Hungarian algorithm on cost matrix (1 - similarity)
    3. Filter assignments below similarity_threshold
    4. Return matched pairs, unmatched extracted (FP), unmatched GT (FN)
    """
    if similarity_model is None:
        from similarity_evaluation.similarity_models import SentenceTransformerSimilarityModel
        similarity_model = SentenceTransformerSimilarityModel("FremyCompany/BioLORD-2023")

    extracted_texts = extracted_df["recommendation"].tolist()
    gt_texts = gt_df["recommendation"].tolist()

    # Handle empty inputs
    if len(extracted_texts) == 0:
        return MatchResult(
            matches=[],
            false_positives=pd.DataFrame(columns=extracted_df.columns),
            false_negatives=gt_df.copy(),
        )
    if len(gt_texts) == 0:
        return MatchResult(
            matches=[],
            false_positives=extracted_df.copy(),
            false_negatives=pd.DataFrame(columns=gt_df.columns),
        )

    # Build similarity matrix and solve assignment
    sim_matrix = build_similarity_matrix(extracted_texts, gt_texts, similarity_model)
    cost_matrix = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build matches, filtering by threshold
    matched_ext_indices = set()
    matched_gt_indices = set()
    matches = []

    for r, c in zip(row_ind, col_ind):
        score = sim_matrix[r, c]
        if score >= similarity_threshold:
            ext_row = extracted_df.iloc[r]
            gt_row = gt_df.iloc[c]
            matches.append(RecommendationMatch(
                extracted_text=ext_row["recommendation"],
                extracted_grade=str(ext_row["class"]),
                extracted_level=str(ext_row["LOE"]),
                gt_text=gt_row["recommendation"],
                gt_grade=str(gt_row["class"]),
                gt_level=str(gt_row["LOE"]),
                similarity_score=score,
            ))
            matched_ext_indices.add(r)
            matched_gt_indices.add(c)

    # Unmatched = false positives / false negatives
    fp_indices = [i for i in range(len(extracted_df)) if i not in matched_ext_indices]
    fn_indices = [i for i in range(len(gt_df)) if i not in matched_gt_indices]

    false_positives = extracted_df.iloc[fp_indices].reset_index(drop=True) if fp_indices else pd.DataFrame(columns=extracted_df.columns)
    false_negatives = gt_df.iloc[fn_indices].reset_index(drop=True) if fn_indices else pd.DataFrame(columns=gt_df.columns)

    return MatchResult(matches=matches, false_positives=false_positives, false_negatives=false_negatives)
