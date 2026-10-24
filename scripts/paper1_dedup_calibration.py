"""Fix the Paper 1 duplicate threshold τ on the 12 development guidelines only (spec §5).

    frozen snapshot recs (dev DOIs) ─▶ BioLORD within-guideline cosine, every pair i < j
    dev labels ─────────────────────▶ each extracted row → its most similar label row (≥ match threshold)
                                       pair is a true duplicate when both rows point to the same label row
    ─▶ pair precision / recall / F1 per τ; the τ with the best F1 goes into configs/paper1.json

Never reads the held-out labels or the rest of the corpus.

Usage:
    python scripts/paper1_dedup_calibration.py --db <data root>/frozen/paper1_snapshot8_2026-10-13.sqlite
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.matching import build_similarity_matrix  # noqa: E402
from evident.extraction import load_similarity_model  # noqa: E402
from evident.store import AccessMode, Store  # noqa: E402
from evident.validation import load_labelled  # noqa: E402
from utils.doi import normalize_doi  # noqa: E402

_SNAPSHOT = 8
_DEVICE = "cpu"
_MATCH_THRESHOLD = 0.65  # evident.validation's label matching threshold
_TAUS = np.round(np.arange(0.80, 0.971, 0.01), 2)  # below the extractor's own 0.97 pass
_NO_LABEL = -1
_SHOW_PAIRS_ABOVE = 0.85


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    args = parser.parse_args()

    labelled = {g.doi: g for g in load_labelled()}
    with Store.open(args.db, AccessMode.READ_ONLY) as store:
        snap = store.get_snapshot(_SNAPSHOT)
        model = load_similarity_model(_DEVICE, store.get_extractor_version(snap.extractor_version_id)
                                      .embedding_revision)
        recs = _dev_recs(store, set(labelled))

    # One (similarity, is true duplicate, doi, text a, text b) per within-guideline pair
    pairs = []
    for doi, texts in sorted(recs.items()):
        label_of = _label_of(texts, [r.text for r in labelled[doi].gt], model)
        sim = build_similarity_matrix(texts, texts, model)
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                same = label_of[i] != _NO_LABEL and label_of[i] == label_of[j]
                pairs.append((float(sim[i, j]), same, doi, texts[i], texts[j]))

    n_true = sum(p[1] for p in pairs)
    print(f"{len(recs)} guidelines, {len(pairs)} pairs, {n_true} true duplicate pairs\n")
    print("tau   pred  tp  precision  recall  f1")
    for tau in _TAUS:
        pred = [p for p in pairs if p[0] >= tau]
        tp = sum(p[1] for p in pred)
        precision = tp / len(pred) if pred else float("nan")
        recall = tp / n_true if n_true else float("nan")
        f1 = 2 * tp / (len(pred) + n_true) if pred or n_true else float("nan")
        print(f"{tau:.2f}  {len(pred):4d} {tp:3d}  {precision:9.3f}  {recall:6.3f}  {f1:.3f}")

    print(f"\nPairs with similarity ≥ {_SHOW_PAIRS_ABOVE} or labelled duplicates:")
    for s, same, doi, a, b in sorted(pairs, key=lambda p: -p[0]):
        if s < _SHOW_PAIRS_ABOVE and not same:
            continue
        print(f"\n{s:.3f} {'DUP ' if same else 'diff'} {doi}\n  A: {a[:200]}\n  B: {b[:200]}")


def _dev_recs(store: Store, dois: set[str]) -> dict[str, list[str]]:
    """{dev doi: texts of the active run in the snapshot}, every row (graded or not)."""
    by_id = {g.id: g for g in store.list_guidelines()}
    out = {}
    for m in store.members(_SNAPSHOT):
        doi = normalize_doi(by_id[m.guideline_id].doi)
        run = store.active_run(_SNAPSHOT, m.guideline_id)
        if doi in dois and run and not m.excluded_reason:
            out[doi] = [r.text for r in store.recommendations(run.id)]
    return out


def _label_of(texts: list[str], gt: list[str], model) -> list[int]:
    """Index of each text's most similar label row, or _NO_LABEL below the match threshold."""
    sim = build_similarity_matrix(texts, gt, model)
    best = sim.argmax(axis=1)
    return [int(b) if sim[i, b] >= _MATCH_THRESHOLD else _NO_LABEL for i, b in enumerate(best)]


if __name__ == "__main__":
    main()
