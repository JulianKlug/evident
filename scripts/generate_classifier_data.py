#!/usr/bin/env python3
"""Generate training data for the recommendation classifier.

For each of the 15 guidelines:
1. Run baseline extraction (qwen3:14b, few-shot, normalize)
2. Match extracted recs to GT using BioLORD at 0.65 threshold
3. Label: matched = 1 (TP), unmatched = 0 (FP)
4. Save CSV + BioLORD embeddings as .npz

Output:
    artifacts/classifier/training_data.csv
    artifacts/classifier/embeddings.npz
"""

from __future__ import annotations

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

OUTPUT_DIR = "artifacts/classifier"
MODEL = "qwen3:14b"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    csv_path = os.path.join(OUTPUT_DIR, "training_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows to {csv_path}", flush=True)
    print(f"  TP: {(df['label'] == 1).sum()}, FP: {(df['label'] == 0).sum()}", flush=True)

    # Compute and save embeddings
    print("Computing BioLORD embeddings...", flush=True)
    texts = df["recommendation"].tolist()
    embeddings = sim_model.encode_batch(texts)
    npz_path = os.path.join(OUTPUT_DIR, "embeddings.npz")
    np.savez_compressed(npz_path, embeddings=embeddings)
    print(f"Saved embeddings ({embeddings.shape}) to {npz_path}", flush=True)

    print("\nDone! Next steps:")
    print("  python -c \"from extraction.recommendation_classifier import train_classifier; train_classifier()\"")


if __name__ == "__main__":
    main()
