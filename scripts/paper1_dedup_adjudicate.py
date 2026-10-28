"""Adjudicate screened duplicate pairs of a snapshot with a local LLM; publish reads the combined verdicts.

    DB + snapshot ─▶ pipeline.duplicate_candidates (BioLORD ≥ 0.75 or lexical ≥ 0.70, within one guideline)
                 ─▶ [--ledger: drop pairs that already have a verdict for their texts]
                 ─▶ LLM per pair: same recommendation restated, or distinct?   (temperature 0, fixed seed)
                 ─▶ one CSV per judge (paper1_dedup_consensus.py combines two of them)

Usage:
    live map (M5), after `add` or a new snapshot; run once per judge, then paper1_dedup_consensus.py:
    python scripts/paper1_dedup_adjudicate.py --db out/evident.sqlite --snapshot ID \
        --ledger corpus/dedup/live_verdicts.csv --model gemma3:27b --out corpus/dedup/judges/live-gemma3-27b.csv
        [--dois DOI ...]        # a subset, e.g. the guideline just added
        [--only-pairs CSV]      # only the pairs listed (doi, ordinal_a, ordinal_b), e.g. judge disagreements
        [--dry-run]             # count the pairs to judge, call no model (budget GPU time first)

    M4 (frozen): --db <data root>/frozen/paper1_snapshot8_2026-10-13.sqlite --snapshot 8, no --ledger
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evident.corpus import load_manifest  # noqa: E402
from evident.dedup import (  # noqa: E402
    JUDGE_MODELS,
    JUDGE_PROMPT,
    JUDGE_SCHEMA,
    Verdict,
    load_adjudications,
    resolve,
    write_adjudications,
)
from evident.domain import SnapshotState  # noqa: E402
from evident.extraction import load_similarity_model  # noqa: E402
from evident.pipeline import duplicate_candidates  # noqa: E402
from evident.store import AccessMode, Store  # noqa: E402
from extraction.llm_client import OllamaClient, ThinkMode  # noqa: E402

_MANIFEST = "corpus/manifest.csv"
_DEVICE = "cpu"  # same similarities as publish --package
_MODEL = JUDGE_MODELS[0]
_SEED = 0
_NUM_CTX = 4096



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--snapshot", type=int, help="snapshot id (default: the published one)")
    parser.add_argument("--manifest", default=_MANIFEST)
    parser.add_argument("--ledger", help="verdict ledger; pairs with a verdict for their texts are skipped")
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--out", required=True, help="this judge's verdicts, e.g. corpus/dedup/judges/live-<model>.csv")
    parser.add_argument("--dois", nargs="*", help="adjudicate only these guidelines")
    parser.add_argument("--only-pairs", help="adjudicate only the pairs in this CSV (doi, ordinal_a, ordinal_b)")
    parser.add_argument("--dry-run", action="store_true", help="print how many pairs need a verdict; no model call")
    args = parser.parse_args()

    with Store.open(args.db, AccessMode.READ_ONLY) as store:
        snap = store.get_snapshot(args.snapshot) if args.snapshot else store.latest_snapshot(SnapshotState.PUBLISHED)
        revision = store.get_extractor_version(snap.extractor_version_id).embedding_revision
        pairs = duplicate_candidates(store, snap.id, args.manifest, load_similarity_model(_DEVICE, revision))
    if args.ledger:
        _, pairs = resolve(pairs, load_adjudications(args.ledger))
    if args.dois:
        pairs = [p for p in pairs if p.doi in set(args.dois)]
    if args.only_pairs:
        wanted = _pair_keys(args.only_pairs)
        pairs = [p for p in pairs if (p.doi, p.ordinal_a, p.ordinal_b) in wanted]

    print(f"snapshot {snap.id}: {len(pairs)} pairs to judge in {len({p.doi for p in pairs})} guidelines", flush=True)
    if args.dry_run:
        return

    titles = {e.doi: e.title for e in load_manifest(args.manifest)}
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
