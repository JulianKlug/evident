#!/usr/bin/env python3
"""Run extraction benchmark with unbuffered output."""

import sys
import os
import time

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

from extraction.benchmark import (
    run_full_benchmark, print_benchmark_summary,
    filter_available_datasets, _BioLORDSimilarityModel, BenchmarkEntry,
    _build_strategy, MODELS, STRATEGIES,
)
from extraction.datasets import load_all_datasets
from extraction.extractor import extract_guideline
from extraction.llm_client import OllamaClient
from evaluation.evaluate import evaluate_extraction

import pandas as pd


def run_benchmark_unbuffered(
    datasets, models, strategies, similarity_model, similarity_threshold=0.65,
    pages_per_chunk=1, output_format="pipe", normalize=False, two_pass=False,
    vision=False, vision_model="gemma3:27b", auto_vision=False,
):
    """Same as run_full_benchmark but with flushed prints and new options."""
    entries = []

    for model_name in models:
        client = OllamaClient(model=model_name)
        for strategy_name in strategies:
            for ds in datasets:
                print(f"[Benchmark] {model_name} / {strategy_name} / {ds.dataset_name}:{ds.key}",
                      flush=True)

                strategy = _build_strategy(strategy_name, ds.grading_scheme, ds.key)

                start = time.time()
                try:
                    if vision:
                        from extraction.vision_extractor import vision_extract_guideline
                        result = vision_extract_guideline(
                            source=ds.doi,
                            strategy=strategy,
                            client=client,
                            vision_model=vision_model,
                            normalize=normalize,
                        )
                    elif auto_vision:
                        from extraction.vision_extractor import auto_vision_extract_guideline
                        result = auto_vision_extract_guideline(
                            source=ds.doi,
                            strategy=strategy,
                            client=client,
                            vision_model=vision_model,
                            pages_per_chunk=pages_per_chunk,
                            output_format=output_format,
                            normalize=normalize,
                        )
                    elif two_pass:
                        from extraction.two_pass import two_pass_extract
                        result = two_pass_extract(
                            source=ds.doi,
                            strategy=strategy,
                            extract_client=client,
                            pages_per_chunk=pages_per_chunk,
                            output_format=output_format,
                            normalize=normalize,
                        )
                    else:
                        result = extract_guideline(
                            source=ds.doi,
                            strategy=strategy,
                            client=client,
                            pages_per_chunk=pages_per_chunk,
                            output_format=output_format,
                            normalize=normalize,
                        )
                except Exception as e:
                    print(f"  ERROR: {e}", flush=True)
                    continue
                extract_elapsed = time.time() - start

                print(f"  Extracted: {result.n_final_recommendations} recs "
                      f"(raw: {result.n_raw_recommendations}) from {result.n_pages} pages "
                      f"in {extract_elapsed:.1f}s", flush=True)

                # Evaluate against ground truth
                eval_start = time.time()
                try:
                    eval_result = evaluate_extraction(
                        extracted_df=result.recommendations_df,
                        gt_df=ds.ground_truth_df,
                        grading_scheme=ds.grading_scheme,
                        similarity_model=similarity_model,
                        similarity_threshold=similarity_threshold,
                    )
                except Exception as e:
                    print(f"  EVAL ERROR: {e}", flush=True)
                    continue
                eval_elapsed = time.time() - eval_start

                entry = BenchmarkEntry(
                    model=model_name,
                    strategy=strategy_name,
                    dataset_name=ds.dataset_name,
                    guideline_key=ds.key,
                    n_gt=len(ds.ground_truth_df),
                    n_extracted=len(result.recommendations_df),
                    precision=eval_result.precision,
                    recall=eval_result.recall,
                    f1=eval_result.f1,
                    grade_accuracy=eval_result.grade_accuracy,
                    level_accuracy=eval_result.level_accuracy,
                    combined_accuracy=eval_result.combined_accuracy,
                    mean_similarity=eval_result.mean_similarity,
                    extraction_time_sec=extract_elapsed,
                )
                entries.append(entry)
                print(f"  P={entry.precision:.2f} R={entry.recall:.2f} F1={entry.f1:.2f} "
                      f"Grade={entry.grade_accuracy:.2f} Level={entry.level_accuracy:.2f} "
                      f"(eval: {eval_elapsed:.1f}s)", flush=True)

                # Save intermediate results
                pd.DataFrame([vars(e) for e in entries]).to_csv(
                    "benchmark_results_partial.csv", index=False
                )

    return pd.DataFrame([vars(e) for e in entries]) if entries else pd.DataFrame()


if __name__ == "__main__":
    print("Loading datasets...", flush=True)
    all_ds = load_all_datasets()
    available = filter_available_datasets(all_ds)

    print(f"Available: {len(available)}/{len(all_ds)} datasets", flush=True)
    for ds in available:
        print(f"  {ds.dataset_name}/{ds.key}: {len(ds.ground_truth_df)} GT recs", flush=True)

    # Parse CLI args for subset
    models = MODELS
    strategies = STRATEGIES
    pages_per_chunk = 1
    output_format = "pipe"
    normalize = False
    two_pass = False

    if "--acp-only" in sys.argv:
        available = [ds for ds in available if ds.dataset_name == "ACP"]
        print(f"Filtering to ACP only: {len(available)} datasets", flush=True)
    if "--ers-only" in sys.argv:
        available = [ds for ds in available if ds.dataset_name == "ERS"]
        print(f"Filtering to ERS only: {len(available)} datasets", flush=True)
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        models = [sys.argv[idx + 1]]
        print(f"Using model: {models[0]}", flush=True)
    if "--zero-shot-only" in sys.argv:
        strategies = ["zero_shot"]
    if "--few-shot-only" in sys.argv:
        strategies = ["few_shot"]
    if "--pages-per-chunk" in sys.argv:
        idx = sys.argv.index("--pages-per-chunk")
        pages_per_chunk = int(sys.argv[idx + 1])
        print(f"Pages per chunk: {pages_per_chunk}", flush=True)
    if "--json" in sys.argv:
        output_format = "json"
        print("Using JSON output format", flush=True)
    if "--normalize" in sys.argv:
        normalize = True
        print("Grade/level normalization enabled", flush=True)
    if "--two-pass" in sys.argv:
        two_pass = True
        print("Two-pass extraction enabled", flush=True)

    vision = False
    auto_vision = False
    vision_model = "gemma3:27b"
    if "--vision" in sys.argv:
        vision = True
        print("Vision-based table extraction enabled", flush=True)
    if "--auto-vision" in sys.argv:
        auto_vision = True
        print("Auto-vision detection enabled", flush=True)
    if "--vision-model" in sys.argv:
        idx = sys.argv.index("--vision-model")
        vision_model = sys.argv[idx + 1]
        print(f"Vision model: {vision_model}", flush=True)

    n_combos = len(available) * len(models) * len(strategies)
    print(f"\nRunning {n_combos} benchmark combinations "
          f"({len(models)} models x {len(strategies)} strategies x {len(available)} guidelines)",
          flush=True)

    print("\nLoading similarity model...", flush=True)
    sim_model = _BioLORDSimilarityModel()
    print("Similarity model ready.\n", flush=True)

    results = run_benchmark_unbuffered(
        available, models, strategies, sim_model,
        pages_per_chunk=pages_per_chunk,
        output_format=output_format,
        normalize=normalize,
        two_pass=two_pass,
        vision=vision,
        vision_model=vision_model,
        auto_vision=auto_vision,
    )

    if not results.empty:
        print_benchmark_summary(results)
        results.to_csv("benchmark_results.csv", index=False)
        print("\nResults saved to benchmark_results.csv", flush=True)
    else:
        print("No results to summarize.", flush=True)
