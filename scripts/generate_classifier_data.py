#!/usr/bin/env python3
"""Generate training data for the recommendation classifier.

For each of the 15 guidelines:
1. Run extraction (baseline or SC Adaptive with --sc flag)
2. Match extracted recs to GT using BioLORD at 0.65 threshold
3. Label: matched = 1 (TP), unmatched = 0 (FP)
4. Save CSV + BioLORD embeddings as .npz

Output:
    artifacts/classifier/training_data.csv          (baseline mode)
    artifacts/classifier_sc/training_data.csv       (--sc mode)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure project root is on sys.path when run from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd

from extraction.benchmark import (
    _BioLORDSimilarityModel, _build_strategy, filter_available_datasets,
)
from extraction.datasets import load_all_datasets
from extraction.extractor import extract_guideline
from extraction.llm_client import OllamaClient
from evaluation.matching import match_recommendations

MODEL = "qwen3:14b"


def main():
    parser = argparse.ArgumentParser(description="Generate classifier training data")
    parser.add_argument("--sc", action="store_true",
                        help="Use SC Adaptive extraction instead of baseline")
    args = parser.parse_args()

    output_dir = "artifacts/classifier_sc" if args.sc else "artifacts/classifier"
    os.makedirs(output_dir, exist_ok=True)

    mode_label = "SC Adaptive" if args.sc else "Baseline"
    print(f"Mode: {mode_label}", flush=True)

    print("Loading datasets...", flush=True)
    all_ds = load_all_datasets()
    available = filter_available_datasets(all_ds)
    print(f"Available: {len(available)}/{len(all_ds)} datasets", flush=True)

    print("Loading similarity model...", flush=True)
    sim_model = _BioLORDSimilarityModel()
    print("Similarity model ready.\n", flush=True)

    client = OllamaClient(model=MODEL)
    all_rows = []

    for i, ds in enumerate(available):
        print(f"[{i+1}/{len(available)}] {ds.dataset_name}:{ds.key} "
              f"({len(ds.ground_truth_df)} GT recs)", flush=True)

        strategy = _build_strategy("few_shot", ds.grading_scheme, ds.key)

        start = time.time()
        try:
            if args.sc:
                from extraction.self_consistency import self_consistency_extract
                result = self_consistency_extract(
                    source=ds.doi,
                    strategy=strategy,
                    client=client,
                    similarity_model=sim_model,
                    n_samples=3,
                    temperature=0.3,
                    adaptive_threshold=True,
                    normalize=True,
                )
            else:
                result = extract_guideline(
                    source=ds.doi,
                    strategy=strategy,
                    client=client,
                    normalize=True,
                )
        except Exception as e:
            print(f"  ERROR extracting: {e}", flush=True)
            continue
        elapsed = time.time() - start
        print(f"  Extracted {len(result.recommendations_df)} recs in {elapsed:.0f}s",
              flush=True)

        ext_df = result.recommendations_df
        if ext_df.empty:
            print("  No extractions, skipping.", flush=True)
            continue

        # Match against GT at 0.65 threshold
        match_result = match_recommendations(
            ext_df, ds.ground_truth_df,
            similarity_model=sim_model,
            similarity_threshold=0.65,
        )

        # Matched extracted indices = TP
        matched_ext_texts = {m.extracted_text for m in match_result.matches}
        n_tp = 0
        n_fp = 0

        for _, row in ext_df.iterrows():
            is_tp = row["recommendation"] in matched_ext_texts
            label = 1 if is_tp else 0
            if is_tp:
                n_tp += 1
            else:
                n_fp += 1
            all_rows.append({
                "recommendation": row["recommendation"],
                "class": row["class"],
                "LOE": row["LOE"],
                "guideline_key": ds.key,
                "dataset_name": ds.dataset_name,
                "label": label,
            })

        print(f"  Labeled: {n_tp} TP, {n_fp} FP", flush=True)

    if not all_rows:
        print("ERROR: No training data generated.", flush=True)
        sys.exit(1)

    # Save CSV
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(output_dir, "training_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows to {csv_path}", flush=True)
    print(f"  TP: {(df['label'] == 1).sum()}, FP: {(df['label'] == 0).sum()}", flush=True)

    # Compute and save embeddings
    print("Computing BioLORD embeddings...", flush=True)
    texts = df["recommendation"].tolist()
    embeddings = sim_model.encode_batch(texts)
    npz_path = os.path.join(output_dir, "embeddings.npz")
    np.savez_compressed(npz_path, embeddings=embeddings)
    print(f"Saved embeddings ({embeddings.shape}) to {npz_path}", flush=True)

    print(f"\nDone! Next steps:")
    print(f"  python -c \"from extraction.recommendation_classifier import train_classifier; "
          f"train_classifier(data_csv='{csv_path}', embeddings_npz='{npz_path}', "
          f"model_output='{output_dir}/rec_classifier.joblib')\"")


if __name__ == "__main__":
    main()
