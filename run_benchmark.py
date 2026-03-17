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
    ensemble=False, ensemble_models=None,
    self_consistency=False, n_samples=3, sc_temperature=0.3, consensus=2,
    verify=False, verify_threshold=0.5,
    ml_filter=False, classifier_path=None, ml_filter_threshold=0.3,
    post_filter=False, filter_model="qwen3:8b",
    adaptive_threshold=False,
    grading_oracle=False,
    oracle_model="deepseek-r1:32b",
    context_oracle=False,
    context_similarity_model=None,
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
                    if self_consistency:
                        from extraction.self_consistency import self_consistency_extract
                        result = self_consistency_extract(
                            source=ds.doi,
                            strategy=strategy,
                            client=client,
                            similarity_model=similarity_model,
                            n_samples=n_samples,
                            temperature=sc_temperature,
                            consensus_threshold=consensus,
                            pages_per_chunk=pages_per_chunk,
                            output_format=output_format,
                            normalize=normalize,
                            adaptive_threshold=adaptive_threshold,
                            ml_filter=ml_filter,
                            classifier_path=classifier_path,
                            ml_similarity_model=similarity_model if ml_filter else None,
                            ml_filter_threshold=ml_filter_threshold,
                            post_filter=post_filter,
                            filter_model=filter_model,
                            grading_oracle=grading_oracle,
                            oracle_model=oracle_model,
                            context_oracle=context_oracle,
                            context_similarity_model=context_similarity_model,
                        )
                    elif ensemble:
                        from extraction.ensemble import ensemble_extract
                        result = ensemble_extract(
                            source=ds.doi,
                            strategy=strategy,
                            models=ensemble_models,
                            pages_per_chunk=pages_per_chunk,
                            output_format=output_format,
                            normalize=normalize,
                        )
                    elif vision:
                        from extraction.vision_extractor import vision_extract_guideline
                        result = vision_extract_guideline(
                            source=ds.doi,
                            strategy=strategy,
                            client=client,
                            vision_model=vision_model,
                            normalize=normalize,
                            verify=verify,
                            verify_threshold=verify_threshold,
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
                            verify=verify,
                            verify_threshold=verify_threshold,
                            ml_filter=ml_filter,
                            classifier_path=classifier_path,
                            ml_similarity_model=similarity_model if ml_filter else None,
                            ml_filter_threshold=ml_filter_threshold,
                            grading_oracle=grading_oracle,
                            oracle_model=oracle_model,
                            context_oracle=context_oracle,
                            context_similarity_model=context_similarity_model,
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
                            verify=verify,
                            verify_threshold=verify_threshold,
                            ml_filter=ml_filter,
                            classifier_path=classifier_path,
                            ml_similarity_model=similarity_model if ml_filter else None,
                            ml_filter_threshold=ml_filter_threshold,
                            post_filter=post_filter,
                            filter_model=filter_model,
                            grading_oracle=grading_oracle,
                            oracle_model=oracle_model,
                            context_oracle=context_oracle,
                            context_similarity_model=context_similarity_model,
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
                    "docs/benchmark_results_partial.csv", index=False
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
    if "--icu-only" in sys.argv:
        available = [ds for ds in available if ds.dataset_name == "ICU"]
        print(f"Filtering to ICU only: {len(available)} datasets", flush=True)
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

    ensemble = False
    ensemble_models = None
    if "--ensemble" in sys.argv:
        ensemble = True
        print("Ensemble extraction enabled", flush=True)
    if "--ensemble-models" in sys.argv:
        idx = sys.argv.index("--ensemble-models")
        ensemble_models = sys.argv[idx + 1].split(",")
        print(f"Ensemble models: {ensemble_models}", flush=True)

    self_consistency = False
    n_samples = 3
    sc_temperature = 0.3
    consensus = 2
    if "--self-consistency" in sys.argv:
        self_consistency = True
        print("Self-consistency voting enabled", flush=True)
    if "--n-samples" in sys.argv:
        idx = sys.argv.index("--n-samples")
        n_samples = int(sys.argv[idx + 1])
        print(f"Self-consistency samples: {n_samples}", flush=True)
    if "--sc-temperature" in sys.argv:
        idx = sys.argv.index("--sc-temperature")
        sc_temperature = float(sys.argv[idx + 1])
        print(f"Self-consistency temperature: {sc_temperature}", flush=True)
    if "--consensus" in sys.argv:
        idx = sys.argv.index("--consensus")
        consensus = int(sys.argv[idx + 1])
        print(f"Consensus threshold: {consensus}", flush=True)

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

    verify = False
    verify_threshold = 0.5
    if "--verify" in sys.argv:
        verify = True
        print("Post-extraction verification enabled", flush=True)
    if "--verify-threshold" in sys.argv:
        idx = sys.argv.index("--verify-threshold")
        verify_threshold = float(sys.argv[idx + 1])
        print(f"Verify threshold: {verify_threshold}", flush=True)

    post_filter = False
    filter_model = "qwen3:8b"
    if "--post-filter" in sys.argv:
        post_filter = True
        print("Post-extraction classification filter enabled", flush=True)
    if "--filter-model" in sys.argv:
        idx = sys.argv.index("--filter-model")
        filter_model = sys.argv[idx + 1]
        print(f"Filter model: {filter_model}", flush=True)

    adaptive_threshold = False
    if "--adaptive-threshold" in sys.argv:
        adaptive_threshold = True
        print("Adaptive self-consistency threshold enabled", flush=True)

    ml_filter = False
    classifier_path = None
    if "--ml-filter" in sys.argv:
        ml_filter = True
        print("ML classification filter enabled", flush=True)
    if "--classifier-path" in sys.argv:
        idx = sys.argv.index("--classifier-path")
        classifier_path = sys.argv[idx + 1]
        print(f"Classifier path: {classifier_path}", flush=True)
    ml_filter_threshold = 0.3
    if "--ml-filter-threshold" in sys.argv:
        idx = sys.argv.index("--ml-filter-threshold")
        ml_filter_threshold = float(sys.argv[idx + 1])
        print(f"ML filter threshold: {ml_filter_threshold}", flush=True)

    grading_oracle = False
    oracle_model = "deepseek-r1:32b"
    context_oracle = False
    if "--grading-oracle" in sys.argv:
        grading_oracle = True
        print("Grading oracle enabled", flush=True)
    if "--context-oracle" in sys.argv:
        context_oracle = True
        print("Context-aware grading oracle enabled", flush=True)
    if "--oracle-model" in sys.argv:
        idx = sys.argv.index("--oracle-model")
        oracle_model = sys.argv[idx + 1]
        print(f"Oracle model: {oracle_model}", flush=True)

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
        ensemble=ensemble,
        ensemble_models=ensemble_models,
        self_consistency=self_consistency,
        n_samples=n_samples,
        sc_temperature=sc_temperature,
        consensus=consensus,
        verify=verify,
        verify_threshold=verify_threshold,
        ml_filter=ml_filter,
        classifier_path=classifier_path,
        ml_filter_threshold=ml_filter_threshold,
        post_filter=post_filter,
        filter_model=filter_model,
        adaptive_threshold=adaptive_threshold,
        grading_oracle=grading_oracle,
        oracle_model=oracle_model,
        context_oracle=context_oracle,
        context_similarity_model=sim_model if context_oracle else None,
    )

    if not results.empty:
        print_benchmark_summary(results)
        results.to_csv("docs/benchmark_results.csv", index=False)
        print("\nResults saved to benchmark_results.csv", flush=True)
    else:
        print("No results to summarize.", flush=True)
