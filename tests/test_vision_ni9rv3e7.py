#!/usr/bin/env python3
"""Test vision extraction on NI9RV3E7."""

import os
import sys
os.environ["PYTHONUNBUFFERED"] = "1"

from extraction.datasets import load_ers_datasets
from extraction.benchmark import _build_strategy
from extraction.vision_extractor import extract_tables_with_vision
from extraction.deduplication import deduplicate_recommendations
from evaluation.evaluate import evaluate_extraction
from extraction.benchmark import _BioLORDSimilarityModel
import pandas as pd

ds = [d for d in load_ers_datasets() if d.key == 'NI9RV3E7'][0]
strategy = _build_strategy('few_shot', ds.grading_scheme, ds.key)

print(f'GT: {len(ds.ground_truth_df)} recs')
print('Extracting from all table page images...', flush=True)

table_dfs = extract_tables_with_vision(ds.doi, strategy, vision_model='gemma3:27b')
total = sum(len(df) for df in table_dfs)
print(f'\nFound {total} raw recs from {len(table_dfs)} pages with tables', flush=True)

if table_dfs:
    combined = pd.concat(table_dfs, ignore_index=True)

    # Deduplicate
    deduped = deduplicate_recommendations(combined, similarity_threshold=0.9)
    print(f'After dedup: {len(deduped)} recs', flush=True)

    deduped.to_csv('/tmp/ni9rv3e7_vision_deduped.csv', index=False)
    print(f'Class distribution: {deduped["class"].value_counts().to_dict()}', flush=True)
    print(f'LOE distribution: {deduped["LOE"].value_counts().to_dict()}', flush=True)

    # Evaluate
    print('\nLoading similarity model...', flush=True)
    sim_model = _BioLORDSimilarityModel()
    eval_result = evaluate_extraction(
        extracted_df=deduped,
        gt_df=ds.ground_truth_df,
        grading_scheme=ds.grading_scheme,
        similarity_model=sim_model,
        similarity_threshold=0.65,
    )
    print(f'\n=== RESULTS ===', flush=True)
    print(f'GT: {len(ds.ground_truth_df)}, Extracted: {len(deduped)}', flush=True)
    print(f'P={eval_result.precision:.3f} R={eval_result.recall:.3f} F1={eval_result.f1:.3f}', flush=True)
    print(f'Grade={eval_result.grade_accuracy:.3f} Level={eval_result.level_accuracy:.3f}', flush=True)
else:
    print('No recommendations extracted!', flush=True)
