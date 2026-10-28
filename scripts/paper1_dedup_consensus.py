"""Combine two duplicate judges into the verdict ledger publish reads: duplicate only when both agree.

    judges/live-qwen38-64k.csv ─┐
    judges/live-gemma3-27b.csv ─┴─▶ consensus (split pair → distinct: in doubt, keep both copies)
                                      ├─▶ judges/disagreements.csv   (the split pairs, for audit)
                                      ▼
                       corpus/dedup/live_verdicts.csv  (merged: existing verdicts are kept, never dropped)

The M4 file corpus/dedup/paper1_verdicts.csv is frozen (hashed in configs/paper1.json): never written here.

Usage:
    python scripts/paper1_dedup_consensus.py --first corpus/dedup/judges/live-qwen38-64k.csv \
        --second corpus/dedup/judges/live-gemma3-27b.csv [--out corpus/dedup/live_verdicts.csv]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evident.dedup import (  # noqa: E402
    MODEL_SEPARATOR,
    CandidatePair,
    DedupVerdictError,
    Verdict,
    consensus,
    disagreements,
    load_adjudications,
    merge_rows,
    read_verdict_rows,
    write_adjudications,
)

_OUT = "corpus/dedup/live_verdicts.csv"
_FROZEN = "corpus/dedup/paper1_verdicts.csv"
_DISAGREEMENTS = "corpus/dedup/judges/disagreements.csv"
_EXIT_MISMATCH = 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--out", default=_OUT)
    parser.add_argument("--disagreements", default=_DISAGREEMENTS)
    args = parser.parse_args()
    if os.path.abspath(args.out) == os.path.abspath(_FROZEN):
        print(f"{_FROZEN} is the frozen M4 file; write the live ledger instead", file=sys.stderr)
        return _EXIT_MISMATCH

    first, second = load_adjudications(args.first), load_adjudications(args.second)
    split = disagreements(first, second)
    pairs = _pairs(args.first)
    _write_disagreements(args.disagreements, split, first, second, pairs)
    print(f"{len(first)} pairs; judges disagree on {len(split)} → {args.disagreements}")

    try:
        verdicts = consensus(first, second)
    except DedupVerdictError as e:
        print(e, file=sys.stderr)
        return _EXIT_MISMATCH

    # Digest column: each judge's digest, in the order of the model field
    digests = {**_digests(args.first), **_digests(args.second)}
    rows = [(pairs[(v.doi, v.ordinal_a, v.ordinal_b)], v.verdict, v.reason, v.model,
             MODEL_SEPARATOR.join(digests[m] for m in v.model.split(MODEL_SEPARATOR)))
            for v in verdicts]

    # Merge into the ledger: earlier verdicts stay (history), a pair judged again must agree with itself
    ledger = read_verdict_rows(args.out) if os.path.isfile(args.out) else []
    try:
        merged = merge_rows(ledger, rows)
    except DedupVerdictError as e:
        print(e, file=sys.stderr)
        return _EXIT_MISMATCH
    write_adjudications(args.out, merged)
    n_dup = sum(v.verdict == Verdict.DUPLICATE for v in verdicts)
    print(f"{len(verdicts)} verdicts, {n_dup} duplicate; ledger {len(ledger)} → {len(merged)} rows → {args.out}")
    return 0


def _pairs(path: str) -> dict[tuple[str, int, int], CandidatePair]:
    """Screened pairs (scores and texts) as written beside a judge's verdicts."""
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["doi"], int(r["ordinal_a"]), int(r["ordinal_b"])):
                CandidatePair(r["doi"], int(r["ordinal_a"]), int(r["ordinal_b"]), float(r["biolord"]),
                              float(r["lexical"]), r["text_a"], r["text_b"])
                for r in csv.DictReader(f)}


def _digests(path: str) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as f:
        return {r["model"]: r["model_digest"] for r in csv.DictReader(f)}


def _write_disagreements(path, keys, first, second, pairs) -> None:
    """The split pairs with both verdicts, for audit and the human check."""
    by_first = {(a.doi, a.ordinal_a, a.ordinal_b): a for a in first}
    by_second = {(a.doi, a.ordinal_a, a.ordinal_b): a for a in second}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["doi", "ordinal_a", "ordinal_b", "first", "second", "text_a", "text_b"])
        for key in keys:
            p = pairs[key]
            w.writerow([*key, by_first[key].verdict.value, by_second[key].verdict.value, p.text_a, p.text_b])


if __name__ == "__main__":
    sys.exit(main())
