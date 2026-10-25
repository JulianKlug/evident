"""Adjudicate screened duplicate pairs of a frozen snapshot with a local LLM; publish reads the frozen verdicts.

    frozen DB ─▶ pipeline.duplicate_candidates (BioLORD ≥ 0.75 or lexical ≥ 0.70, within one guideline)
              ─▶ LLM per pair: same recommendation restated, or distinct?   (temperature 0, fixed seed)
              ─▶ corpus/dedup/judges/<model>.csv   (one file per judge; paper1_dedup_consensus.py combines them)

Usage:
    python scripts/paper1_dedup_adjudicate.py --db <data root>/frozen/paper1_snapshot8_2026-10-13.sqlite \
        --model gemma3:27b --out corpus/dedup/judges/gemma3-27b.csv
        [--dois DOI ...]        # a subset, e.g. the development guidelines
        [--only-pairs CSV]      # only the pairs listed (doi, ordinal_a, ordinal_b), e.g. judge disagreements
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evident.corpus import load_manifest  # noqa: E402
from evident.dedup import JUDGE_PROMPT, JUDGE_SCHEMA, Verdict, write_adjudications  # noqa: E402
from evident.extraction import load_similarity_model  # noqa: E402
from evident.pipeline import duplicate_candidates  # noqa: E402
from evident.store import AccessMode, Store  # noqa: E402
from extraction.llm_client import OllamaClient, ThinkMode  # noqa: E402

_SNAPSHOT = 8
_MANIFEST = "corpus/manifest.csv"
_OUT = "corpus/dedup/paper1_verdicts.csv"
_DEVICE = "cpu"  # same similarities as publish --package
_MODEL = "qwen38-64k:latest"
_SEED = 0
_NUM_CTX = 4096



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--out", default=_OUT)
    parser.add_argument("--dois", nargs="*", help="adjudicate only these guidelines (a trial run)")
    parser.add_argument("--only-pairs", help="adjudicate only the pairs in this CSV (doi, ordinal_a, ordinal_b)")
    args = parser.parse_args()

    with Store.open(args.db, AccessMode.READ_ONLY) as store:
        snap = store.get_snapshot(_SNAPSHOT)
        revision = store.get_extractor_version(snap.extractor_version_id).embedding_revision
        pairs = duplicate_candidates(store, _SNAPSHOT, _MANIFEST, load_similarity_model(_DEVICE, revision))
    if args.dois:
        pairs = [p for p in pairs if p.doi in set(args.dois)]
    if args.only_pairs:
        wanted = _pair_keys(args.only_pairs)
        pairs = [p for p in pairs if (p.doi, p.ordinal_a, p.ordinal_b) in wanted]

    titles = {e.doi: e.title for e in load_manifest(_MANIFEST)}
    client = OllamaClient(args.model)
    digest = client.model_digest()
    rows = []
    for i, p in enumerate(pairs, start=1):
        duplicate, reason = _ask(client, titles[p.doi], p.text_a, p.text_b)
        rows.append((p, Verdict.DUPLICATE if duplicate else Verdict.DISTINCT, reason, args.model, digest))
        print(f"{i}/{len(pairs)} {p.doi} {p.ordinal_a}-{p.ordinal_b} "
              f"bio {p.biolord:.2f} lex {p.lexical:.2f} → {rows[-1][1].value}: {reason}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_adjudications(args.out, rows)
    n_dup = sum(r[1] == Verdict.DUPLICATE for r in rows)
    print(f"{len(rows)} pairs, {n_dup} duplicate → {args.out}")


def _pair_keys(path: str) -> set[tuple[str, int, int]]:
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["doi"], int(r["ordinal_a"]), int(r["ordinal_b"])) for r in csv.DictReader(f)}


def _ask(client: OllamaClient, title: str, a: str, b: str) -> tuple[bool, str]:
    """(duplicate, reason); an unparseable answer is distinct, so nothing is removed on a model failure."""
    response = client.generate_json(JUDGE_PROMPT.format(title=title, a=a, b=b), JUDGE_SCHEMA, num_ctx=_NUM_CTX, seed=_SEED,
                                    think=ThinkMode.OFF)
    try:
        answer = json.loads(response.raw_text)
    except json.JSONDecodeError:
        return False, "unparseable answer"
    return bool(answer.get("duplicate")), " ".join(str(answer.get("reason", "")).split())


if __name__ == "__main__":
    main()
