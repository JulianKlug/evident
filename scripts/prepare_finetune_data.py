#!/usr/bin/env python3
"""Generate page-level fine-tuning data from ground truth + PDFs.

For each guideline: load PDF pages, assign each GT recommendation to the page
where it appears (token-level Jaccard similarity), then create ChatML training
examples:
  - Positive: system prompt + page text -> pipe-delimited GT recs on that page
  - Negative: system prompt + page text -> NO_RECOMMENDATIONS_FOUND
  - Downsample negatives to 2x positives per guideline

Output: JSONL with ChatML messages for Unsloth/TRL SFTTrainer.

Usage:
    python scripts/prepare_finetune_data.py --output-dir artifacts/finetune
    python scripts/prepare_finetune_data.py --output-dir artifacts/finetune --logo-cv
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extraction.datasets import load_all_datasets, GuidelineDataset
from extraction.benchmark import filter_available_datasets
from extraction.pdf_loader import load_pdf_pages, PDFPage
from extraction.prompts import _SCHEME_TERMINOLOGY
from evaluation.grading import GradingScheme


def _tokenize(text: str) -> Set[str]:
    """Simple whitespace + lowercase tokenization for Jaccard similarity."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _containment(rec_tokens: Set[str], page_tokens: Set[str]) -> float:
    """Fraction of recommendation tokens found in the page.

    Better than Jaccard for short recs vs long pages — measures how much
    of the recommendation text appears in the page, not vice versa.
    """
    if not rec_tokens:
        return 0.0
    return len(rec_tokens & page_tokens) / len(rec_tokens)


def assign_recs_to_pages(
    pages: List[PDFPage],
    gt_df: pd.DataFrame,
    threshold: float = 0.5,
) -> Dict[int, List[int]]:
    """Assign each GT recommendation to its best-matching page.

    Uses token containment (fraction of rec tokens found in page) rather
    than Jaccard, since recommendations are short relative to full pages.

    Args:
        pages: PDF pages with text.
        gt_df: Ground truth with 'recommendation' column.
        threshold: Minimum containment score to assign a rec to a page.

    Returns:
        Tuple of (page_assignments dict, unmatched list).
    """
    page_tokens = [_tokenize(p.text) for p in pages]
    page_assignments = defaultdict(list)
    unmatched = []

    for rec_idx, row in gt_df.iterrows():
        rec_text = str(row["recommendation"])
        rec_tokens = _tokenize(rec_text)

        best_page = -1
        best_score = 0.0

        for page_idx, p_tokens in enumerate(page_tokens):
            score = _containment(rec_tokens, p_tokens)
            if score > best_score:
                best_score = score
                best_page = page_idx

        if best_score >= threshold and best_page >= 0:
            page_assignments[best_page].append(rec_idx)
        else:
            unmatched.append((rec_idx, rec_text[:80], best_score))

    return dict(page_assignments), unmatched


def build_system_prompt(scheme: GradingScheme) -> str:
    """Build the system prompt for a given grading scheme (no few-shot examples)."""
    terms = _SCHEME_TERMINOLOGY[scheme.name]
    grade_label = terms["grade_label"]
    level_label = terms["level_label"]
    grade_values = terms["grade_values"]
    level_values = terms["level_values"]

    return (
        f"You are an expert medical researcher. Extract all clinical recommendations "
        f"from the following {terms['domain']} text.\n"
        f"\n"
        f"A \"recommendation\" is an actionable statement that directs clinical practice "
        f"and is explicitly graded with a {grade_label} and a {level_label}.\n"
        f"\n"
        f"For each recommendation, extract:\n"
        f"- The recommendation text — copy it EXACTLY as written in the source\n"
        f"- The {grade_label} — use ONLY these valid values: {grade_values}\n"
        f"- The {level_label} — use ONLY these valid values: {level_values}\n"
        f"\n"
        f"Output format: one recommendation per line, using pipe delimiters:\n"
        f"recommendation text | {grade_label} | {level_label}\n"
        f"\n"
        f"Rules:\n"
        f"- Extract ONLY explicitly stated recommendations with a clear {grade_label} and {level_label}\n"
        f"- Copy the recommendation text EXACTLY as written — do NOT paraphrase or summarize\n"
        f"- Do NOT extract: background statements, evidence summaries, section headers, "
        f"or statements without an explicit {grade_label} and {level_label}\n"
        f"- Do NOT infer or create recommendations that are not in the text\n"
        f"- If no recommendations are found, output exactly: NO_RECOMMENDATIONS_FOUND\n"
        f"- Do NOT include headers, row numbers, or any other text"
    )


def build_user_message(page_text: str) -> str:
    """Build the user message containing the guideline text."""
    return (
        f"--- Guideline Text ---\n{page_text}\n--- End of Text ---\n\n"
        f"Extracted recommendations:"
    )


def build_assistant_response(gt_df: pd.DataFrame, rec_indices: List[int]) -> str:
    """Build the expected assistant response from GT recs."""
    if not rec_indices:
        return "NO_RECOMMENDATIONS_FOUND"

    lines = []
    for idx in rec_indices:
        row = gt_df.iloc[idx]
        rec = str(row["recommendation"]).strip()
        grade = str(row["class"]).strip()
        level = str(row["LOE"]).strip()
        lines.append(f"{rec} | {grade} | {level}")

    return "\n".join(lines)


def create_training_examples(
    ds: GuidelineDataset,
    pages: List[PDFPage],
    max_negatives_ratio: float = 2.0,
    threshold: float = 0.5,
    max_examples_per_guideline: int = 50,
    seed: int = 42,
) -> Tuple[List[dict], dict]:
    """Create ChatML training examples for one guideline.

    Returns:
        Tuple of (examples_list, stats_dict)
    """
    page_assignments, unmatched = assign_recs_to_pages(
        pages, ds.ground_truth_df, threshold=threshold,
    )

    system_prompt = build_system_prompt(ds.grading_scheme)

    positive_examples = []
    negative_examples = []

    for page_idx, page in enumerate(pages):
        rec_indices = page_assignments.get(page_idx, [])
        user_msg = build_user_message(page.text)
        assistant_msg = build_assistant_response(ds.ground_truth_df, rec_indices)

        example = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            "metadata": {
                "guideline_key": ds.key,
                "dataset_name": ds.dataset_name,
                "scheme": ds.grading_scheme.name,
                "page_number": page.page_number,
                "n_recs": len(rec_indices),
            },
        }

        if rec_indices:
            positive_examples.append(example)
        else:
            negative_examples.append(example)

    # Downsample negatives
    n_positives = len(positive_examples)
    max_negatives = int(n_positives * max_negatives_ratio)
    rng = random.Random(seed + hash(ds.key))
    if len(negative_examples) > max_negatives:
        negative_examples = rng.sample(negative_examples, max_negatives)

    all_examples = positive_examples + negative_examples

    # Cap total per guideline
    if len(all_examples) > max_examples_per_guideline:
        # Keep all positives, downsample negatives further
        if len(positive_examples) >= max_examples_per_guideline:
            all_examples = rng.sample(positive_examples, max_examples_per_guideline)
        else:
            remaining = max_examples_per_guideline - len(positive_examples)
            all_examples = positive_examples + rng.sample(
                negative_examples, min(remaining, len(negative_examples))
            )

    n_assigned = sum(len(v) for v in page_assignments.values())
    stats = {
        "guideline_key": ds.key,
        "dataset_name": ds.dataset_name,
        "scheme": ds.grading_scheme.name,
        "n_gt_recs": len(ds.ground_truth_df),
        "n_pages": len(pages),
        "n_assigned": n_assigned,
        "n_unmatched": len(unmatched),
        "n_positive_pages": n_positives,
        "n_negative_pages": len(negative_examples),
        "n_total_examples": len(all_examples),
        "unmatched_recs": unmatched[:5],  # sample for debugging
    }

    return all_examples, stats


def write_jsonl(examples: List[dict], path: str):
    """Write examples to JSONL file (only the messages, no metadata)."""
    with open(path, "w") as f:
        for ex in examples:
            # Write only the messages (what the trainer needs)
            record = {"messages": ex["messages"]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare fine-tuning data from GT + PDFs")
    parser.add_argument("--output-dir", default="artifacts/finetune",
                        help="Output directory for training data")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Min token containment for page assignment")
    parser.add_argument("--max-neg-ratio", type=float, default=2.0,
                        help="Max negatives per positive per guideline")
    parser.add_argument("--max-per-guideline", type=int, default=50,
                        help="Max examples per guideline")
    parser.add_argument("--logo-cv", action="store_true",
                        help="Also generate leave-one-guideline-out CV splits")
    parser.add_argument("--upsample", action="store_true",
                        help="Upsample underrepresented schemes to balance training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading datasets...", flush=True)
    all_ds = load_all_datasets()
    available = filter_available_datasets(all_ds)
    print(f"Available: {len(available)}/{len(all_ds)} datasets\n", flush=True)

    all_examples = []
    all_stats = []

    for ds in available:
        print(f"Processing {ds.dataset_name}/{ds.key} "
              f"({len(ds.ground_truth_df)} GT recs)...", end="  ", flush=True)

        pages = load_pdf_pages(ds.doi)
        examples, stats = create_training_examples(
            ds, pages,
            max_negatives_ratio=args.max_neg_ratio,
            threshold=args.threshold,
            max_examples_per_guideline=args.max_per_guideline,
            seed=args.seed,
        )

        all_examples.extend(examples)
        all_stats.append(stats)

        print(f"{stats['n_positive_pages']}+ / {stats['n_negative_pages']}- pages, "
              f"{stats['n_assigned']}/{stats['n_gt_recs']} recs assigned, "
              f"{stats['n_unmatched']} unmatched", flush=True)

    # Upsample underrepresented schemes to balance training
    if args.upsample:
        scheme_examples = defaultdict(list)
        for ex in all_examples:
            scheme_examples[ex["metadata"]["scheme"]].append(ex)

        # Find the majority scheme count
        scheme_counts_raw = {s: len(exs) for s, exs in scheme_examples.items()}
        max_count = max(scheme_counts_raw.values())
        print(f"\nScheme balancing (target ~{max_count} per scheme):")

        upsampled = []
        rng_up = random.Random(args.seed + 1)
        for scheme, exs in sorted(scheme_examples.items()):
            n_orig = len(exs)
            if n_orig >= max_count:
                upsampled.extend(exs)
                print(f"  {scheme}: {n_orig} (no change)")
            else:
                # Repeat + sample remainder to reach target
                n_needed = max_count - n_orig
                repeats = n_needed // n_orig
                remainder = n_needed % n_orig
                expanded = exs * (1 + repeats) + rng_up.sample(exs, remainder)
                upsampled.extend(expanded)
                print(f"  {scheme}: {n_orig} -> {len(expanded)} ({repeats+1}x + {remainder})")

        all_examples = upsampled
        print(f"Total after upsampling: {len(all_examples)}")

    # Shuffle all examples
    rng = random.Random(args.seed)
    rng.shuffle(all_examples)

    # Write full training data
    train_path = os.path.join(args.output_dir, "train_data.jsonl")
    write_jsonl(all_examples, train_path)
    print(f"\nWrote {len(all_examples)} examples to {train_path}")

    # Write stats report
    stats_path = os.path.join(args.output_dir, "data_stats.json")
    total_pos = sum(s["n_positive_pages"] for s in all_stats)
    total_neg = sum(s["n_negative_pages"] for s in all_stats)
    total_assigned = sum(s["n_assigned"] for s in all_stats)
    total_gt = sum(s["n_gt_recs"] for s in all_stats)
    total_unmatched = sum(s["n_unmatched"] for s in all_stats)

    summary = {
        "total_examples": len(all_examples),
        "total_positive_pages": total_pos,
        "total_negative_pages": total_neg,
        "total_gt_recs": total_gt,
        "total_assigned_recs": total_assigned,
        "total_unmatched_recs": total_unmatched,
        "assignment_rate": total_assigned / total_gt if total_gt else 0,
        "per_guideline": all_stats,
    }
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Stats written to {stats_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total examples:     {len(all_examples)}")
    print(f"  Positive pages:   {total_pos}")
    print(f"  Negative pages:   {total_neg}")
    print(f"GT assignment rate: {total_assigned}/{total_gt} "
          f"({100*total_assigned/total_gt:.1f}%)")
    print(f"Unmatched recs:     {total_unmatched}")

    # Per-scheme breakdown
    scheme_counts = defaultdict(lambda: {"examples": 0, "recs": 0})
    for s in all_stats:
        scheme_counts[s["scheme"]]["examples"] += s["n_total_examples"]
        scheme_counts[s["scheme"]]["recs"] += s["n_gt_recs"]
    print(f"\nPer-scheme:")
    for scheme, counts in sorted(scheme_counts.items()):
        print(f"  {scheme}: {counts['examples']} examples, {counts['recs']} GT recs")

    # LOGO-CV splits
    if args.logo_cv:
        print(f"\nGenerating LOGO-CV splits...", flush=True)
        cv_dir = os.path.join(args.output_dir, "logo_cv")
        os.makedirs(cv_dir, exist_ok=True)

        guideline_keys = [s["guideline_key"] for s in all_stats]
        for held_out_key in guideline_keys:
            fold_examples = [
                ex for ex in all_examples
                if ex["metadata"]["guideline_key"] != held_out_key
            ]
            fold_path = os.path.join(cv_dir, f"train_excl_{held_out_key}.jsonl")
            write_jsonl(fold_examples, fold_path)
            print(f"  {held_out_key}: {len(fold_examples)} train examples "
                  f"(held out {len(all_examples) - len(fold_examples)})")

    print("\nDone!")


if __name__ == "__main__":
    main()
