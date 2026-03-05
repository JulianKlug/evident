"""Two-pass extraction: classify pages first, then extract from relevant pages only."""

from __future__ import annotations

from typing import TYPE_CHECKING

from extraction.llm_client import OllamaClient
from extraction.pdf_loader import load_pdf_pages, PDFPage
from extraction.prompts import PromptStrategy
from extraction.extractor import extract_guideline, ExtractionResult, _chunk_pages

if TYPE_CHECKING:
    from similarity_evaluation.similarity_models import SimilarityModel


_CLASSIFICATION_PROMPT = (
    "Does the following page from a clinical practice guideline contain "
    "any formal clinical recommendations with explicit grading "
    "(e.g., strength of recommendation and level of evidence)?\n"
    "\n"
    "Answer YES or NO only.\n"
    "\n"
    "--- Page Text ---\n"
    "{page_text}\n"
    "--- End of Page ---\n"
    "\n"
    "Answer:"
)


def classify_pages(
    pages: list[PDFPage],
    client: OllamaClient,
) -> list[bool]:
    """Classify each page as containing recommendations (True) or not (False).

    Uses a fast, simple prompt to get YES/NO answers.
    """
    results = []
    for page in pages:
        prompt = _CLASSIFICATION_PROMPT.format(page_text=page.text)
        response = client.generate(prompt)
        answer = response.raw_text.strip().upper()
        results.append(answer.startswith("YES"))
    return results


def two_pass_extract(
    source: str,
    strategy: PromptStrategy,
    classify_client: OllamaClient | None = None,
    extract_client: OllamaClient | None = None,
    dedup_threshold: float = 0.9,
    dedup_model: "SimilarityModel | None" = None,
    pages_per_chunk: int = 3,
    output_format: str = "pipe",
    normalize: bool = False,
    context_pages: int = 1,
) -> ExtractionResult:
    """Two-pass extraction pipeline.

    Pass 1 (classify_client): Classify each page as YES/NO for containing recommendations.
    Pass 2 (extract_client): Extract only from YES pages with multi-page chunking,
        including surrounding context pages.

    Args:
        source: Local PDF path or DOI string.
        strategy: Prompting strategy.
        classify_client: Client for pass 1 (fast model). Defaults to llama3.2:latest.
        extract_client: Client for pass 2 (quality model). Defaults to strategy's model.
        dedup_threshold: Similarity threshold for deduplication.
        dedup_model: Optional similarity model for semantic dedup.
        pages_per_chunk: Pages per chunk for pass 2.
        output_format: "pipe" or "json".
        normalize: If True, normalize grades/levels post-extraction.
        context_pages: Number of surrounding pages to include around YES pages.

    Returns:
        ExtractionResult with recommendations from relevant pages only.
    """
    if classify_client is None:
        classify_client = OllamaClient(model="llama3.2:latest")
    if extract_client is None:
        extract_client = OllamaClient()

    pages = load_pdf_pages(source)

    # Pass 1: Classify pages
    page_has_recs = classify_pages(pages, classify_client)

    # Expand selection to include context pages around YES pages
    selected = set()
    for i, has_recs in enumerate(page_has_recs):
        if has_recs:
            for j in range(max(0, i - context_pages), min(len(pages), i + context_pages + 1)):
                selected.add(j)

    relevant_pages = [pages[i] for i in sorted(selected)]

    if not relevant_pages:
        import pandas as pd
        from extraction.llm_client import LLMResponse
        return ExtractionResult(
            recommendations_df=pd.DataFrame(columns=["recommendation", "class", "LOE"]),
            n_pages=len(pages),
            n_pages_with_recs=0,
            n_raw_recommendations=0,
            n_final_recommendations=0,
            per_page_responses=[],
        )

    # Pass 2: Extract from relevant pages using the full pipeline
    # We create a temporary PDF-like source by writing relevant pages
    # Instead, we directly use the extraction logic on the filtered pages
    from extraction.extractor import (
        _chunk_pages, _EXTRACTION_SCHEMA,
        deduplicate_recommendations,
    )
    from extraction.prompts import build_prompt
    from extraction.response_parser import parse_llm_response, parse_json_response
    import pandas as pd

    chunks = _chunk_pages(relevant_pages, pages_per_chunk=pages_per_chunk)

    all_dfs = []
    responses = []
    n_chunks_with_recs = 0

    for chunk_text in chunks:
        prompt = build_prompt(chunk_text, strategy, output_format=output_format)

        if output_format == "json":
            response = extract_client.generate_json(prompt, schema=_EXTRACTION_SCHEMA)
        else:
            response = extract_client.generate(prompt)
        responses.append(response)

        if output_format == "json":
            chunk_df = parse_json_response(response.raw_text)
        else:
            chunk_df = parse_llm_response(response.raw_text, has_thinking=extract_client.has_thinking)

        if not chunk_df.empty:
            n_chunks_with_recs += 1
            all_dfs.append(chunk_df)

    if all_dfs:
        raw_df = pd.concat(all_dfs, ignore_index=True)
    else:
        raw_df = pd.DataFrame(columns=["recommendation", "class", "LOE"])

    n_raw = len(raw_df)

    final_df = deduplicate_recommendations(
        raw_df,
        similarity_threshold=dedup_threshold,
        similarity_model=dedup_model,
    )

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
