"""Core extraction pipeline: PDF → pages → LLM → parse → dedup → DataFrame."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from extraction.pdf_loader import load_pdf_pages, PDFPage
from extraction.prompts import PromptStrategy, build_prompt
from extraction.llm_client import OllamaClient, LLMResponse
from extraction.response_parser import parse_llm_response, parse_json_response
from extraction.deduplication import deduplicate_recommendations

if TYPE_CHECKING:
    from similarity_evaluation.similarity_models import SimilarityModel


# JSON schema for structured extraction output
_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "grade": {"type": "string"},
                    "level": {"type": "string"},
                },
                "required": ["text", "grade", "level"],
            },
        },
    },
    "required": ["recommendations"],
}


@dataclass
class ExtractionResult:
    """Result of extracting recommendations from a guideline PDF."""
    recommendations_df: pd.DataFrame  # columns: recommendation, class, LOE
    n_pages: int
    n_pages_with_recs: int
    n_raw_recommendations: int
    n_final_recommendations: int
    per_page_responses: list[LLMResponse] = field(default_factory=list)
    vision_pages: list = field(default_factory=list)  # pages processed by vision


def _chunk_pages(
    pages: list[PDFPage],
    pages_per_chunk: int = 3,
    overlap: int = 1,
) -> list[str]:
    """Group pages into overlapping chunks, concatenating their text.

    Args:
        pages: List of PDFPage objects.
        pages_per_chunk: Number of pages per chunk.
        overlap: Number of pages to overlap between consecutive chunks.

    Returns:
        List of concatenated text chunks.
    """
    if pages_per_chunk <= 0:
        raise ValueError("pages_per_chunk must be positive")

    if pages_per_chunk == 1:
        return [p.text for p in pages]

    chunks = []
    step = max(1, pages_per_chunk - overlap)
    for start in range(0, len(pages), step):
        end = min(start + pages_per_chunk, len(pages))
        chunk_pages = pages[start:end]
        chunk_text = "\n--- Page Break ---\n".join(p.text for p in chunk_pages)
        chunks.append(chunk_text)
        if end >= len(pages):
            break

    return chunks


def extract_guideline(
    source: str,
    strategy: PromptStrategy,
    client: OllamaClient | None = None,
    dedup_threshold: float = 0.9,
    dedup_model: SimilarityModel | None = None,
    pages_per_chunk: int = 1,
    output_format: str = "pipe",
    normalize: bool = False,
) -> ExtractionResult:
    """Extract recommendations from a guideline PDF.

    Pipeline: load PDF → chunk pages → for each chunk: build prompt → LLM generate → parse → concat → dedup.

    Args:
        source: Local PDF path or DOI string.
        strategy: Prompting strategy (zero_shot or few_shot with scheme).
        client: OllamaClient instance (created with default model if None).
        dedup_threshold: Similarity threshold for semantic deduplication.
        dedup_model: Optional similarity model for semantic dedup.
        pages_per_chunk: Number of pages per LLM call (1 = original per-page behavior).
        output_format: "pipe" for pipe-delimited or "json" for structured JSON output.
        normalize: If True, normalize extracted grades/levels against the grading scheme.

    Returns:
        ExtractionResult with deduplicated recommendations and metadata.
    """
    if client is None:
        client = OllamaClient()

    pages = load_pdf_pages(source)
    chunks = _chunk_pages(pages, pages_per_chunk=pages_per_chunk)

    all_dfs = []
    responses = []
    n_chunks_with_recs = 0

    # Only set explicit num_ctx when multi-page chunks need more context
    ctx_override = None
    if pages_per_chunk > 1:
        ctx_override = client.model_info.get("context_window", 8192)

    for chunk_text in chunks:
        prompt = build_prompt(chunk_text, strategy, output_format=output_format)

        if output_format == "json":
            response = client.generate_json(prompt, schema=_EXTRACTION_SCHEMA, num_ctx=ctx_override)
        else:
            response = client.generate(prompt, num_ctx=ctx_override)
        responses.append(response)

        if output_format == "json":
            chunk_df = parse_json_response(response.raw_text)
        else:
            chunk_df = parse_llm_response(response.raw_text, has_thinking=client.has_thinking)

        if not chunk_df.empty:
            n_chunks_with_recs += 1
            all_dfs.append(chunk_df)

    if all_dfs:
        raw_df = pd.concat(all_dfs, ignore_index=True)
    else:
        raw_df = pd.DataFrame(columns=["recommendation", "class", "LOE"])

    n_raw = len(raw_df)

    # Deduplicate across chunks
    final_df = deduplicate_recommendations(
        raw_df,
        similarity_threshold=dedup_threshold,
        similarity_model=dedup_model,
    )

    # Post-extraction normalization
    if normalize and not final_df.empty:
        from extraction.postprocessing import normalize_extracted_grades
        final_df = normalize_extracted_grades(final_df, strategy.scheme)

    return ExtractionResult(
        recommendations_df=final_df,
        n_pages=len(pages),
        n_pages_with_recs=n_chunks_with_recs,
        n_raw_recommendations=n_raw,
        n_final_recommendations=len(final_df),
        per_page_responses=responses,
    )
