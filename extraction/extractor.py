"""Core extraction pipeline: PDF → pages → LLM → parse → dedup → DataFrame."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from extraction.pdf_loader import load_pdf_pages
from extraction.prompts import PromptStrategy, build_prompt
from extraction.llm_client import OllamaClient, LLMResponse
from extraction.response_parser import parse_llm_response
from extraction.deduplication import deduplicate_recommendations

if TYPE_CHECKING:
    from similarity_evaluation.similarity_models import SimilarityModel


@dataclass
class ExtractionResult:
    """Result of extracting recommendations from a guideline PDF."""
    recommendations_df: pd.DataFrame  # columns: recommendation, class, LOE
    n_pages: int
    n_pages_with_recs: int
    n_raw_recommendations: int
    n_final_recommendations: int
    per_page_responses: list[LLMResponse] = field(default_factory=list)


def extract_guideline(
    source: str,
    strategy: PromptStrategy,
    client: OllamaClient | None = None,
    dedup_threshold: float = 0.9,
    dedup_model: SimilarityModel | None = None,
) -> ExtractionResult:
    """Extract recommendations from a guideline PDF.

    Pipeline: load PDF → for each page: build prompt → LLM generate → parse → concat → dedup.

    Args:
        source: Local PDF path or DOI string.
        strategy: Prompting strategy (zero_shot or few_shot with scheme).
        client: OllamaClient instance (created with default model if None).
        dedup_threshold: Similarity threshold for semantic deduplication.
        dedup_model: Optional similarity model for semantic dedup.

    Returns:
        ExtractionResult with deduplicated recommendations and metadata.
    """
    if client is None:
        client = OllamaClient()

    pages = load_pdf_pages(source)

    all_dfs = []
    responses = []
    n_pages_with_recs = 0

    for page in pages:
        prompt = build_prompt(page.text, strategy)
        response = client.generate(prompt)
        responses.append(response)

        page_df = parse_llm_response(response.raw_text, has_thinking=client.has_thinking)
        if not page_df.empty:
            n_pages_with_recs += 1
            all_dfs.append(page_df)

    if all_dfs:
        raw_df = pd.concat(all_dfs, ignore_index=True)
    else:
        raw_df = pd.DataFrame(columns=["recommendation", "class", "LOE"])

    n_raw = len(raw_df)

    # Deduplicate across pages
    final_df = deduplicate_recommendations(
        raw_df,
        similarity_threshold=dedup_threshold,
        similarity_model=dedup_model,
    )

    return ExtractionResult(
        recommendations_df=final_df,
        n_pages=len(pages),
        n_pages_with_recs=n_pages_with_recs,
        n_raw_recommendations=n_raw,
        n_final_recommendations=len(final_df),
        per_page_responses=responses,
    )
