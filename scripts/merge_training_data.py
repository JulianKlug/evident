#!/usr/bin/env python3
"""Merge real and synthetic training data for fine-tuning.

Combines the original page-level training data with synthetic augmented data,
with optional scheme balancing.

Usage:
    python scripts/merge_training_data.py \
        --real artifacts/finetune/train_data.jsonl \
        --synthetic artifacts/finetune/synthetic/synthetic_data.jsonl \
        --output artifacts/finetune/train_data_augmented.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict


def load_jsonl(path: str):
    """Load JSONL records."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def detect_scheme(messages: list) -> str:
    """Detect grading scheme from system prompt content."""
    system = messages[0].get("content", "") if messages else ""
    if "class of recommendation" in system:
        return "esc_ers"
    elif "grade of recommendation" in system:
        return "abcd_123"
    else:
        return "grade"


def main():
    parser = argparse.ArgumentParser(description="Merge real + synthetic training data")
    parser.add_argument("--real", required=True, help="Path to real training JSONL")
    parser.add_argument("--synthetic", required=True, help="Path to synthetic JSONL")
    parser.add_argument("--output", required=True, help="Output merged JSONL path")
    parser.add_argument("--max-synthetic-ratio", type=float, default=3.0,
                        help="Max ratio of synthetic:real examples per scheme")
    parser.add_argument("--duplicate-positives", type=int, default=1,
                        help="Duplicate positive examples N times (N=2 means each positive appears twice)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load data
    real = load_jsonl(args.real)
    synthetic = load_jsonl(args.synthetic)
    print(f"Real examples: {len(real)}")
    print(f"Synthetic examples: {len(synthetic)}")

    # Group by scheme
    real_by_scheme = defaultdict(list)
    for r in real:
        scheme = detect_scheme(r["messages"])
        real_by_scheme[scheme].append(r)

    synth_by_scheme = defaultdict(list)
    for s in synthetic:
        scheme = detect_scheme(s["messages"])
        synth_by_scheme[scheme].append(s)

    # Cap synthetic per scheme based on ratio to real
    merged = []
    print(f"\nPer-scheme breakdown:")
    for scheme in sorted(set(list(real_by_scheme.keys()) + list(synth_by_scheme.keys()))):
        n_real = len(real_by_scheme[scheme])
        n_synth_available = len(synth_by_scheme[scheme])
        max_synth = int(n_real * args.max_synthetic_ratio)
        n_synth_used = min(n_synth_available, max_synth)

        merged.extend(real_by_scheme[scheme])
        if n_synth_used < n_synth_available:
            merged.extend(rng.sample(synth_by_scheme[scheme], n_synth_used))
        else:
            merged.extend(synth_by_scheme[scheme])

        print(f"  {scheme}: {n_real} real + {n_synth_used} synthetic "
              f"(of {n_synth_available} available) = {n_real + n_synth_used}")

    # Duplicate positive examples for loss weighting
    if args.duplicate_positives > 1:
        positives = []
        for record in merged:
            msgs = record.get("messages", [])
            # Check if assistant response is not NO_RECOMMENDATIONS_FOUND
            assistant_msg = ""
            for m in msgs:
                if m.get("role") == "assistant":
                    assistant_msg = m.get("content", "")
            if assistant_msg.strip() != "NO_RECOMMENDATIONS_FOUND":
                positives.append(record)

        n_dupes = len(positives) * (args.duplicate_positives - 1)
        for _ in range(args.duplicate_positives - 1):
            merged.extend(positives)
        print(f"\nPositive duplication ({args.duplicate_positives}x): "
              f"added {n_dupes} copies of {len(positives)} positive examples")

    # Shuffle
    rng.shuffle(merged)

    # Write
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for record in merged:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nTotal merged: {len(merged)}")
    print(f"Written to: {args.output}")


if __name__ == "__main__":
    main()
