"""Core extraction pipeline: PDF → pages → LLM → parse → dedup → DataFrame."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import pandas as pd

from extraction.pdf_loader import load_pdf_pages, PDFPage
from extraction.prompts import PromptStrategy, build_prompt
from extraction.llm_client import OllamaClient, LLMResponse, ThinkMode
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
                "additionalProperties": False,
            },
        },
    },
    "required": ["recommendations"],
}


@dataclass
class CallRecord:
    """One LLM call with what is needed to reproduce it."""
    sample_idx: int  # 0 for single pass; SC sample index otherwise
    chunk_id: int
    seed: int | None
    prompt_sha256: str
    response: LLMResponse


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _think_kwargs(think: ThinkMode) -> dict:
    # Kwarg omitted when DEFAULT, so legacy clients without `think` keep working
    return {} if think == ThinkMode.DEFAULT else {"think": think}


@dataclass
class ExtractionResult:
    """Result of extracting recommendations from a guideline PDF."""
    recommendations_df: pd.DataFrame  # columns: recommendation, class, LOE, category, page, chunk_id
    n_pages: int
    n_pages_with_recs: int
    n_raw_recommendations: int
    n_final_recommendations: int
    per_page_responses: list[LLMResponse] = field(default_factory=list)
    vision_pages: list = field(default_factory=list)  # pages processed by vision
    call_records: list[CallRecord] = field(default_factory=list)


def _chunk_spans(
    pages: list[PDFPage],
    pages_per_chunk: int = 3,
    overlap: int = 1,
) -> list[tuple[int, str]]:
    """Group pages into overlapping chunks, keeping each chunk's 1-based start page.

    Example: 3 pages, pages_per_chunk=2, overlap=1 → [(1, "p1 … p2"), (2, "p2 … p3")].
    """
    if pages_per_chunk <= 0:
        raise ValueError("pages_per_chunk must be positive")

    if pages_per_chunk == 1:
        return [(p.page_number, p.text) for p in pages]

    spans = []
    step = max(1, pages_per_chunk - overlap)
    for start in range(0, len(pages), step):
        end = min(start + pages_per_chunk, len(pages))
        chunk_pages = pages[start:end]
        chunk_text = "\n--- Page Break ---\n".join(p.text for p in chunk_pages)
        spans.append((chunk_pages[0].page_number, chunk_text))
        if end >= len(pages):
            break

    return spans


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
    return [text for _, text in _chunk_spans(pages, pages_per_chunk, overlap)]


def extract_guideline(
    source: str,
    strategy: PromptStrategy,
    client: OllamaClient | None = None,
    dedup_threshold: float = 0.9,
    dedup_model: SimilarityModel | None = None,
    pages_per_chunk: int = 1,
    output_format: str = "pipe",
    normalize: bool = False,
    verify: bool = False,
    verify_threshold: float = 0.5,
    ml_filter: bool = False,
    classifier_path: Optional[str] = None,
    ml_similarity_model: SimilarityModel | None = None,
    ml_filter_threshold: float = 0.3,
    post_filter: bool = False,
    filter_model: str = "qwen3:8b",
    grading_oracle: bool = False,
    oracle_model: str = "deepseek-r1:32b",
    context_oracle: bool = False,
    context_similarity_model: SimilarityModel | None = None,
    seed: int | None = None,
    think: ThinkMode = ThinkMode.DEFAULT,
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
        verify: If True, filter out recommendations not grounded in source text.
        verify_threshold: Minimum token overlap fraction for verification (default 0.5).
        ml_filter: If True, apply ML-based classification filter using BioLORD + LR.
        classifier_path: Path to trained classifier (default artifacts/classifier/rec_classifier.joblib).
        ml_similarity_model: Pre-loaded BioLORD model for embeddings (avoids double-loading).
        post_filter: If True, apply binary classification filter to remove non-recommendations.
        filter_model: Model to use for post-filter classification (default qwen3:8b).
        grading_oracle: If True, re-grade recommendations using a reasoning model.
        oracle_model: Model to use for grading oracle (default deepseek-r1:32b).
        context_oracle: If True, re-grade using context-aware oracle with BioLORD retrieval.
        context_similarity_model: Pre-loaded BioLORD model for context retrieval.
        seed: Sampling seed, forwarded to the client only when set.
        think: Thinking switch, forwarded to the client only when not DEFAULT.

    Returns:
        ExtractionResult with deduplicated recommendations and metadata.
    """
    if grading_oracle and context_oracle:
        raise ValueError("Cannot use both --grading-oracle and --context-oracle")

    if client is None:
        client = OllamaClient()

    pages = load_pdf_pages(source)
    spans = _chunk_spans(pages, pages_per_chunk=pages_per_chunk)
    chunks = [text for _, text in spans]

    all_dfs = []
    responses = []
    call_records = []
    seed_kwargs = {} if seed is None else {"seed": seed}
    call_kwargs = {**seed_kwargs, **_think_kwargs(think)}
    n_chunks_with_recs = 0

    # Only set explicit num_ctx when multi-page chunks need more context
    ctx_override = None
    if pages_per_chunk > 1:
        ctx_override = client.model_info.get("context_window", 8192)

    for chunk_id, (start_page, chunk_text) in enumerate(spans):
        prompt = build_prompt(chunk_text, strategy, output_format=output_format)

        if output_format == "json":
            response = client.generate_json(prompt, schema=_EXTRACTION_SCHEMA, num_ctx=ctx_override, **call_kwargs)
        else:
            response = client.generate(prompt, num_ctx=ctx_override, **call_kwargs)
        responses.append(response)
        call_records.append(CallRecord(0, chunk_id, seed, _prompt_sha256(prompt), response))

        if output_format == "json":
            chunk_df = parse_json_response(response.raw_text)
        else:
            chunk_df = parse_llm_response(response.raw_text, has_thinking=client.has_thinking)

        if not chunk_df.empty:
            # Provenance: chunk's start page and index
            chunk_df["page"] = start_page
            chunk_df["chunk_id"] = chunk_id
            n_chunks_with_recs += 1
            all_dfs.append(chunk_df)

    if all_dfs:
        raw_df = pd.concat(all_dfs, ignore_index=True)
    else:
        raw_df = pd.DataFrame(columns=["recommendation", "class", "LOE", "category", "page", "chunk_id"])

    n_raw = len(raw_df)

    # Deduplicate across chunks
    final_df = deduplicate_recommendations(
        raw_df,
        similarity_threshold=dedup_threshold,
        similarity_model=dedup_model,
    )

    # Post-extraction verification
    if verify and not final_df.empty:
        from extraction.verification import verify_recommendations
        final_df = verify_recommendations(
            final_df, chunks, min_token_overlap=verify_threshold,
        )

    # ML-based classification filter
    if ml_filter and not final_df.empty:
        from extraction.recommendation_classifier import ml_filter_recommendations, load_classifier
        from extraction.benchmark import _BioLORDSimilarityModel
        _clf = load_classifier(classifier_path or "artifacts/classifier/rec_classifier.joblib")
        _sim = ml_similarity_model or _BioLORDSimilarityModel()
        final_df = ml_filter_recommendations(final_df, _clf, _sim, threshold=ml_filter_threshold)

    # Post-extraction classification filter
    if post_filter and not final_df.empty:
        from extraction.classification_filter import classify_recommendations
        filter_client = OllamaClient(model=filter_model)
        final_df = classify_recommendations(final_df, filter_client, strategy.scheme)

    # Post-extraction normalization
    if normalize and not final_df.empty:
        from extraction.postprocessing import normalize_extracted_grades
        final_df = normalize_extracted_grades(final_df, strategy.scheme)

    # Post-extraction grading oracle
    if grading_oracle and not final_df.empty:
        from extraction.grading_oracle import regrade_recommendations
        final_df = regrade_recommendations(final_df, strategy.scheme, model=oracle_model)

    # Context-aware grading oracle
    if context_oracle and not final_df.empty:
        from extraction.grading_oracle import regrade_with_context
        from extraction.benchmark import _BioLORDSimilarityModel
        _ctx_sim = context_similarity_model or _BioLORDSimilarityModel()
        page_texts = [p.text for p in pages]
        final_df = regrade_with_context(
            final_df, strategy.scheme, page_texts,
            similarity_model=_ctx_sim, model=oracle_model,
        )

    return ExtractionResult(
        recommendations_df=final_df,
        n_pages=len(pages),
        n_pages_with_recs=n_chunks_with_recs,
        n_raw_recommendations=n_raw,
        n_final_recommendations=len(final_df),
        per_page_responses=responses,
        call_records=call_records,
    )
