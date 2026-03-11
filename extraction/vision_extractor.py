"""Vision-based table extraction using a multimodal LLM.

For PDFs where table cell text is not extractable as text (e.g., vector-rendered
tables), this module renders table regions as images and uses a vision-capable
LLM to extract recommendations.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from extraction.pdf_loader import (
    load_pdf_pages, load_pdf_table_images, detect_opaque_tables,
    PDFTableImage,
)
from extraction.prompts import PromptStrategy
from extraction.llm_client import OllamaClient
from extraction.deduplication import deduplicate_recommendations
from extraction.extractor import ExtractionResult


_VISION_PROMPT = """\
Look at this page from a clinical guideline. If there is a table with clinical recommendations, extract each recommendation.
Each table row contains: Recommendation text, Class ({grade_label}), Level ({level_label}).

Rules:
- First, count the number of visible recommendation rows in the table
- Extract ONLY rows that are VISIBLE in the table — do NOT generate or infer recommendations that are not shown
- If multiple recommendations share a Class or Level cell, list each separately
- Use the EXACT Class and Level values shown in the table
- Do NOT extract table headers, footnotes, or running text outside the table
- Do NOT paraphrase or abbreviate — copy the recommendation text as written

Output one recommendation per line as: recommendation text | Class | Level

If this page has NO recommendation table (e.g., only definitions, classifications, prognostic factors, or abbreviations), respond: NONE
"""


def _build_vision_prompt(strategy: PromptStrategy) -> str:
    """Build the vision extraction prompt with scheme-specific labels."""
    scheme = strategy.scheme
    if scheme:
        grade_label = "e.g. " + ", ".join(scheme.grades[:4])
        level_label = "e.g. " + ", ".join(scheme.levels[:4])
    else:
        grade_label = "grade"
        level_label = "level"
    return _VISION_PROMPT.format(grade_label=grade_label, level_label=level_label)


def _parse_vision_response(text: str) -> pd.DataFrame:
    """Parse pipe-delimited recommendations from vision model response."""
    rows = []
    for line in text.split("\n"):
        line = line.strip()
        # Remove markdown list markers
        line = re.sub(r"^\*\s+", "", line)
        line = re.sub(r"^-\s+", "", line)
        line = re.sub(r"^\d+\.\s+", "", line)

        if "|" not in line:
            continue
        if line.upper().strip() == "NONE":
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue

        rec_text = parts[0]
        grade = parts[1]
        level = parts[2]

        # Skip header-like rows
        if grade.lower() in ("class", "class*", "grade", ""):
            continue
        # Skip if rec text is too short or looks like a label
        if len(rec_text) < 10:
            continue
        # Remove bold markdown
        rec_text = rec_text.replace("**", "").strip()
        grade = grade.replace("**", "").strip()
        level = level.replace("**", "").strip()
        # Remove label prefixes the model sometimes adds
        for prefix in ("Recommendation text: ", "Class: ", "Level: "):
            rec_text = rec_text.removeprefix(prefix) if hasattr(rec_text, 'removeprefix') else rec_text
            grade = grade.removeprefix(prefix) if hasattr(grade, 'removeprefix') else grade
            level = level.removeprefix(prefix) if hasattr(level, 'removeprefix') else level

        if rec_text and grade and level:
            rows.append({
                "recommendation": rec_text,
                "class": grade,
                "LOE": level,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["recommendation", "class", "LOE"])


def extract_tables_with_vision(
    source: str,
    strategy: PromptStrategy,
    vision_model: str = "gemma3:27b",
    resolution: int = 300,
    table_images: Optional[list] = None,
) -> list[pd.DataFrame]:
    """Extract recommendations from PDF table images using a vision model.

    Args:
        source: PDF path or DOI.
        strategy: PromptStrategy with grading scheme info.
        vision_model: Ollama model name with vision capability.
        resolution: DPI for rendering table images.
        table_images: Pre-loaded PDFTableImage list. If provided, skips loading.

    Returns:
        List of DataFrames, one per table that yielded recommendations.
    """
    import ollama

    if table_images is None:
        table_images = load_pdf_table_images(source, resolution=resolution)
    if not table_images:
        return []

    prompt = _build_vision_prompt(strategy)
    results = []

    for timg in table_images:
        response = ollama.generate(
            model=vision_model,
            prompt=prompt,
            images=[timg.image_bytes],
            options={
                "temperature": 0,
                "num_ctx": 4096,
                "top_p": 0.1,
                "repeat_penalty": 1.1,
            },
        )

        df = _parse_vision_response(response["response"])
        if not df.empty:
            results.append(df)

    return results


def vision_extract_guideline(
    source: str,
    strategy: PromptStrategy,
    client: Optional[OllamaClient] = None,
    vision_model: str = "gemma3:27b",
    dedup_threshold: float = 0.9,
    dedup_model=None,
    normalize: bool = False,
    text_extraction: bool = True,
    verify: bool = False,
    verify_threshold: float = 0.5,
) -> ExtractionResult:
    """Extract recommendations using both text and vision-based table extraction.

    Combines standard text-based page extraction with vision-based table extraction
    for PDFs with non-extractable table content.

    Args:
        source: PDF path or DOI.
        strategy: PromptStrategy with grading scheme.
        client: OllamaClient for text-based extraction (if text_extraction=True).
        vision_model: Ollama vision model for table images.
        dedup_threshold: Similarity threshold for deduplication.
        dedup_model: Optional similarity model for dedup.
        normalize: Apply grade/level normalization.
        text_extraction: Also run standard text-based extraction and merge.

    Returns:
        ExtractionResult with combined recommendations.
    """
    all_dfs = []
    n_pages = 0
    responses = []

    # Vision-based table extraction
    print(f"  [Vision] Extracting tables with {vision_model}...", flush=True)
    table_dfs = extract_tables_with_vision(
        source, strategy, vision_model=vision_model,
    )
    n_table_recs = sum(len(df) for df in table_dfs)
    print(f"  [Vision] Found {n_table_recs} recs from {len(table_dfs)} tables", flush=True)
    all_dfs.extend(table_dfs)

    # Optional text-based extraction
    if text_extraction:
        from extraction.extractor import extract_guideline
        text_result = extract_guideline(
            source, strategy, client=client,
            normalize=False,  # normalize after merge
            verify=verify,
            verify_threshold=verify_threshold,
        )
        n_pages = text_result.n_pages
        responses = text_result.per_page_responses
        if not text_result.recommendations_df.empty:
            all_dfs.append(text_result.recommendations_df)
            print(f"  [Text] Found {len(text_result.recommendations_df)} recs from text",
                  flush=True)

    if all_dfs:
        raw_df = pd.concat(all_dfs, ignore_index=True)
    else:
        raw_df = pd.DataFrame(columns=["recommendation", "class", "LOE"])

    n_raw = len(raw_df)

    # Deduplicate across vision + text results
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
        n_pages=n_pages,
        n_pages_with_recs=len(table_dfs),
        n_raw_recommendations=n_raw,
        n_final_recommendations=len(final_df),
        per_page_responses=responses,
    )


def auto_vision_extract_guideline(
    source: str,
    strategy: PromptStrategy,
    client: Optional[OllamaClient] = None,
    vision_model: str = "gemma3:27b",
    dedup_threshold: float = 0.9,
    dedup_model=None,
    pages_per_chunk: int = 1,
    output_format: str = "pipe",
    normalize: bool = False,
    verify: bool = False,
    verify_threshold: float = 0.5,
) -> ExtractionResult:
    """Extract recommendations, auto-detecting when vision is needed.

    Scans the PDF for pages with table structures but no extractable text.
    If found, runs vision extraction on those pages and text extraction on
    the rest, then merges and deduplicates.

    Args:
        source: PDF path or DOI.
        strategy: PromptStrategy with grading scheme.
        client: OllamaClient for text-based extraction.
        vision_model: Ollama vision model for table images.
        dedup_threshold: Similarity threshold for deduplication.
        dedup_model: Optional similarity model for dedup.
        pages_per_chunk: Pages per chunk for text extraction.
        output_format: "pipe" or "json" for text extraction.
        normalize: Apply grade/level normalization.

    Returns:
        ExtractionResult with combined recommendations.
    """
    from extraction.extractor import extract_guideline

    # Detect if vision is needed
    report = detect_opaque_tables(source)

    if not report.needs_vision:
        print("  [Auto-vision] No opaque tables detected, using text extraction only",
              flush=True)
        return extract_guideline(
            source, strategy, client=client,
            pages_per_chunk=pages_per_chunk,
            output_format=output_format,
            normalize=normalize,
            verify=verify,
            verify_threshold=verify_threshold,
        )

    print(f"  [Auto-vision] Detected {len(report.opaque_table_pages)} opaque table pages: "
          f"{report.opaque_table_pages[:10]}{'...' if len(report.opaque_table_pages) > 10 else ''}",
          flush=True)

    all_dfs = []
    n_pages = 0
    responses = []
    vision_pages = []

    # Text extraction (handles text-extractable pages automatically)
    text_result = extract_guideline(
        source, strategy, client=client,
        pages_per_chunk=pages_per_chunk,
        output_format=output_format,
        normalize=False,  # normalize after merge
        verify=verify,
        verify_threshold=verify_threshold,
    )
    n_pages = text_result.n_pages
    responses = text_result.per_page_responses
    if not text_result.recommendations_df.empty:
        all_dfs.append(text_result.recommendations_df)
        print(f"  [Auto-vision] Text extraction: {len(text_result.recommendations_df)} recs",
              flush=True)

    # Vision extraction on opaque table pages only
    print(f"  [Auto-vision] Running vision on {len(report.opaque_table_pages)} pages "
          f"with {vision_model}...", flush=True)
    images = load_pdf_table_images(
        source, page_numbers=report.opaque_table_pages,
    )
    table_dfs = extract_tables_with_vision(
        source, strategy, vision_model=vision_model, table_images=images,
    )
    n_vision_recs = sum(len(df) for df in table_dfs)
    print(f"  [Auto-vision] Vision extraction: {n_vision_recs} recs from "
          f"{len(table_dfs)} pages", flush=True)
    all_dfs.extend(table_dfs)
    vision_pages = report.opaque_table_pages

    # Merge and deduplicate
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
        n_pages=n_pages + len(report.opaque_table_pages),
        n_pages_with_recs=text_result.n_pages_with_recs + len(table_dfs),
        n_raw_recommendations=n_raw,
        n_final_recommendations=len(final_df),
        per_page_responses=responses,
        vision_pages=vision_pages,
    )
