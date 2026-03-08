"""Ensemble extraction: run multiple models and merge+dedup their extractions."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import pandas as pd

from extraction.extractor import extract_guideline, ExtractionResult
from extraction.deduplication import deduplicate_recommendations
from extraction.llm_client import OllamaClient
from extraction.prompts import PromptStrategy

if TYPE_CHECKING:
    from similarity_evaluation.similarity_models import SimilarityModel


# Default ensemble: best F1 + best grade accuracy
DEFAULT_ENSEMBLE_MODELS = ["qwen3:14b", "deepseek-r1:32b"]


def ensemble_extract(
    source: str,
    strategy: PromptStrategy,
    models: Optional[list[str]] = None,
    dedup_threshold: float = 0.9,
    dedup_model: "SimilarityModel | None" = None,
    pages_per_chunk: int = 1,
    output_format: str = "pipe",
    normalize: bool = False,
) -> ExtractionResult:
    """Run multiple models on the same input and merge+dedup their extractions.

    This takes the union of recommendations from all models and deduplicates,
    which should improve recall at the potential cost of precision.

    Args:
        source: Local PDF path or DOI string.
        strategy: Prompting strategy.
        models: List of model names. Defaults to DEFAULT_ENSEMBLE_MODELS.
        dedup_threshold: Similarity threshold for deduplication.
        dedup_model: Optional similarity model for semantic dedup.
        pages_per_chunk: Number of pages per LLM call.
        output_format: "pipe" or "json".
        normalize: If True, normalize grades/levels post-extraction.

    Returns:
        ExtractionResult with merged and deduplicated recommendations.
    """
    if models is None:
        models = DEFAULT_ENSEMBLE_MODELS

    all_dfs = []
    all_responses = []
    n_pages = 0
    n_pages_with_recs = 0

    for model_name in models:
        print(f"  [Ensemble] Running {model_name}...", flush=True)
        client = OllamaClient(model=model_name)
        result = extract_guideline(
            source=source,
            strategy=strategy,
            client=client,
            pages_per_chunk=pages_per_chunk,
            output_format=output_format,
            normalize=False,  # normalize after merge
        )
        n_pages = result.n_pages  # same across models
        n_pages_with_recs += result.n_pages_with_recs
        all_responses.extend(result.per_page_responses)

        print(f"  [Ensemble] {model_name}: {result.n_final_recommendations} recs "
              f"(raw: {result.n_raw_recommendations})", flush=True)

        if not result.recommendations_df.empty:
            all_dfs.append(result.recommendations_df)

    if all_dfs:
        raw_df = pd.concat(all_dfs, ignore_index=True)
    else:
        raw_df = pd.DataFrame(columns=["recommendation", "class", "LOE"])

    n_raw = len(raw_df)
    print(f"  [Ensemble] Merged: {n_raw} recs from {len(models)} models", flush=True)

    # Deduplicate across all model outputs
    final_df = deduplicate_recommendations(
        raw_df,
        similarity_threshold=dedup_threshold,
        similarity_model=dedup_model,
    )

    if normalize and not final_df.empty:
        from extraction.postprocessing import normalize_extracted_grades
        final_df = normalize_extracted_grades(final_df, strategy.scheme)

    print(f"  [Ensemble] After dedup: {len(final_df)} recs", flush=True)

    return ExtractionResult(
        recommendations_df=final_df,
        n_pages=n_pages,
        n_pages_with_recs=n_pages_with_recs,
        n_raw_recommendations=n_raw,
        n_final_recommendations=len(final_df),
        per_page_responses=all_responses,
    )
