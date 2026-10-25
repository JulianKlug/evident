"""Combine the duplicate judges into the frozen verdicts publish reads: majority of three.

    judges/qwen38-64k.csv ─┐
    judges/gemma3-27b.csv ─┴─▶ disagreements ─▶ judges/disagreements.csv ─▶ third judge (adjudicate --only-pairs)
                                    │                                                │
                                    └──────── consensus (agree, or third decides) ◀──┘
                                                   ▼
                                    corpus/dedup/paper1_verdicts.csv  (then its SHA-256 into configs/paper1.json)

Usage:
    python scripts/paper1_dedup_consensus.py --first corpus/dedup/judges/qwen38-64k.csv \
        --second corpus/dedup/judges/gemma3-27b.csv [--third corpus/dedup/judges/mistral-small3.2-24b.csv]

Without --third (or with one missing a disagreement) it writes the disagreements and stops.
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
    write_adjudications,
)

_OUT = "corpus/dedup/paper1_verdicts.csv"
_DISAGREEMENTS = "corpus/dedup/judges/disagreements.csv"
_EXIT_NEEDS_THIRD = 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--third", help="verdicts on the disagreements only")
    parser.add_argument("--out", default=_OUT)
    parser.add_argument("--disagreements", default=_DISAGREEMENTS)
    args = parser.parse_args()

    first, second = load_adjudications(args.first), load_adjudications(args.second)
    split = disagreements(first, second)
    pairs = _pairs(args.first)
    _write_disagreements(args.disagreements, split, first, second, pairs)
    print(f"{len(first)} pairs; judges disagree on {len(split)} → {args.disagreements}")

    third = load_adjudications(args.third) if args.third else []
    try:
        verdicts = consensus(first, second, third)
    except DedupVerdictError as e:
        print(e, file=sys.stderr)
        return _EXIT_NEEDS_THIRD

    # Digest column: each judge's digest, in the order of the model field
    digests = {**_digests(args.first), **_digests(args.second), **(_digests(args.third) if args.third else {})}
    rows = [(pairs[(v.doi, v.ordinal_a, v.ordinal_b)], v.verdict, v.reason, v.model,
             MODEL_SEPARATOR.join(digests[m] for m in v.model.split(MODEL_SEPARATOR)))
            for v in verdicts]
    write_adjudications(args.out, rows)
    n_dup = sum(v.verdict == Verdict.DUPLICATE for v in verdicts)
    print(f"{len(verdicts)} verdicts, {n_dup} duplicate → {args.out}")
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
    """The split pairs with both verdicts: input to the third judge (--only-pairs) and to the human check."""
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
