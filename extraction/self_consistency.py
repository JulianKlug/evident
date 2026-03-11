"""Self-consistency voting: run extraction N times with temperature > 0, keep consensus recs."""

from __future__ import annotations

from collections import Counter
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from extraction.extractor import ExtractionResult, _chunk_pages
from extraction.pdf_loader import load_pdf_pages
from extraction.prompts import PromptStrategy, build_prompt
from extraction.llm_client import OllamaClient
from extraction.response_parser import parse_llm_response
from extraction.deduplication import deduplicate_recommendations

if TYPE_CHECKING:
    pass


def _cluster_recommendations(
    texts: list[str],
    similarity_threshold: float,
    encode_fn,
) -> list[list[int]]:
    """Cluster recommendation texts by semantic similarity using greedy assignment.

    Args:
        texts: List of recommendation texts.
        similarity_threshold: Minimum cosine similarity to group together.
        encode_fn: Function that takes list[str] and returns np.ndarray of embeddings.

    Returns:
        List of clusters, where each cluster is a list of indices into `texts`.
    """
    if not texts:
        return []

    embeddings = encode_fn(texts)  # (N, D), L2-normalized
    sim_matrix = embeddings @ embeddings.T  # cosine similarity

    n = len(texts)
    assigned = [False] * n
    clusters = []

    for i in range(n):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            if sim_matrix[i, j] >= similarity_threshold:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    return clusters


def _majority_vote(values: list[str]) -> str:
    """Return the most common non-empty value, or empty string."""
    non_empty = [v for v in values if v and str(v).strip()]
    if not non_empty:
        return ""
    counts = Counter(non_empty)
    return counts.most_common(1)[0][0]


def self_consistency_extract(
    source: str,
    strategy: PromptStrategy,
    client: OllamaClient,
    similarity_model,
    n_samples: int = 3,
    temperature: float = 0.3,
    consensus_threshold: int = 2,
    similarity_threshold: float = 0.85,
    pages_per_chunk: int = 1,
    output_format: str = "pipe",
    normalize: bool = False,
) -> ExtractionResult:
    """Extract recommendations using self-consistency voting.

    Runs the extraction N times with temperature > 0, clusters the pooled
    recommendations by semantic similarity, and keeps only those appearing
    in >= consensus_threshold samples.

    Args:
        source: Local PDF path or DOI string.
        strategy: Prompting strategy.
        client: OllamaClient instance.
        similarity_model: Model with encode_batch() for clustering.
        n_samples: Number of stochastic extraction runs.
        temperature: Sampling temperature for each run.
        consensus_threshold: Minimum number of samples a rec must appear in.
        similarity_threshold: Cosine similarity threshold for clustering recs.
        pages_per_chunk: Number of pages per LLM call.
        output_format: "pipe" for pipe-delimited output.
        normalize: If True, normalize grades/levels post-extraction.

    Returns:
        ExtractionResult with consensus-filtered recommendations.
    """
    pages = load_pdf_pages(source)
    chunks = _chunk_pages(pages, pages_per_chunk=pages_per_chunk)

    ctx_override = None
    if pages_per_chunk > 1:
        ctx_override = client.model_info.get("context_window", 8192)

    # Collect recs from N stochastic runs
    all_recs = []  # list of (sample_idx, recommendation, class, LOE)
    all_responses = []
    total_raw = 0

    for sample_idx in range(n_samples):
        print(f"  [SC] Sample {sample_idx + 1}/{n_samples}...", flush=True)
        sample_dfs = []

        for chunk_text in chunks:
            prompt = build_prompt(chunk_text, strategy, output_format=output_format)
            response = client.generate(prompt, num_ctx=ctx_override, temperature=temperature)
            all_responses.append(response)

            chunk_df = parse_llm_response(response.raw_text, has_thinking=client.has_thinking)
            if not chunk_df.empty:
                sample_dfs.append(chunk_df)

        if sample_dfs:
            sample_df = pd.concat(sample_dfs, ignore_index=True)
            # Dedup within this sample (exact + semantic)
            sample_df = deduplicate_recommendations(sample_df, similarity_threshold=0.9)
            total_raw += len(sample_df)
            for _, row in sample_df.iterrows():
                all_recs.append((
                    sample_idx,
                    row["recommendation"],
                    row.get("class", ""),
                    row.get("LOE", ""),
                ))
            print(f"  [SC] Sample {sample_idx + 1}: {len(sample_df)} recs (deduped)", flush=True)
        else:
            print(f"  [SC] Sample {sample_idx + 1}: 0 recs", flush=True)

    if not all_recs:
        empty_df = pd.DataFrame(columns=["recommendation", "class", "LOE"])
        return ExtractionResult(
            recommendations_df=empty_df,
            n_pages=len(pages),
            n_pages_with_recs=0,
            n_raw_recommendations=0,
            n_final_recommendations=0,
            per_page_responses=all_responses,
        )

    # Build pooled dataframe with sample tags
    pooled_df = pd.DataFrame(all_recs, columns=["sample_idx", "recommendation", "class", "LOE"])
    texts = pooled_df["recommendation"].tolist()

    # Cluster by semantic similarity
    clusters = _cluster_recommendations(
        texts,
        similarity_threshold=similarity_threshold,
        encode_fn=similarity_model.encode_batch,
    )

    # Filter by consensus and pick representatives
    kept_rows = []
    for cluster_indices in clusters:
        sample_indices = set(pooled_df.iloc[cluster_indices]["sample_idx"])
        n_votes = len(sample_indices)

        if n_votes < consensus_threshold:
            continue

        # Pick representative: longest text in the cluster
        cluster_rows = pooled_df.iloc[cluster_indices]
        best_idx = cluster_rows["recommendation"].str.len().idxmax()
        best_row = cluster_rows.loc[best_idx]

        # Majority vote for grade and level
        grade = _majority_vote(cluster_rows["class"].tolist())
        level = _majority_vote(cluster_rows["LOE"].tolist())

        kept_rows.append({
            "recommendation": best_row["recommendation"],
            "class": grade,
            "LOE": level,
        })

    if kept_rows:
        final_df = pd.DataFrame(kept_rows)
    else:
        final_df = pd.DataFrame(columns=["recommendation", "class", "LOE"])

    print(f"  [SC] Consensus: {len(final_df)} recs from {len(clusters)} clusters "
          f"(threshold={consensus_threshold}/{n_samples})", flush=True)

    # Post-extraction normalization
    if normalize and not final_df.empty:
        from extraction.postprocessing import normalize_extracted_grades
        final_df = normalize_extracted_grades(final_df, strategy.scheme)

    return ExtractionResult(
        recommendations_df=final_df,
        n_pages=len(pages),
        n_pages_with_recs=0,  # not tracked per-sample
        n_raw_recommendations=total_raw,
        n_final_recommendations=len(final_df),
        per_page_responses=all_responses,
    )
