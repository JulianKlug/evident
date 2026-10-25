"""Draw the 15-pair human check of the duplicate verdicts, and score it once labelled.

A check, not an accuracy estimate: 0 errors in 15 still allows up to 22% (95% Clopper–Pearson).
Table S2 (no duplicate removal) bounds the effect of wrong merges; this sample looks where errors matter:

    out/paper1_v2/duplicates.csv ─┬─ 5 removed copies whose grade differs from the kept copy  (grade enters tables)
                                  └─ 5 removed copies with the same grade
    judges/disagreements.csv ────── 5 pairs the first two judges split (the third decided)
                                  ▼ fixed seed, shuffled, verdicts hidden
    corpus/dedup/audit/human_sample.csv  (label: same / different)  +  human_key.csv

Usage:
    python scripts/paper1_dedup_human_sample.py draw  [--duplicates out/paper1_v2/duplicates.csv]
    python scripts/paper1_dedup_human_sample.py score
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys

from scipy.stats import beta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evident.dedup import Verdict, load_adjudications  # noqa: E402

_SEED = 0
_N_GRADE_CONFLICT = 5
_N_SAME_GRADE = 5
_N_SPLIT = 5
_CONFIDENCE = 0.95
_DUPLICATES = "out/paper1_v2/duplicates.csv"
_DISAGREEMENTS = "corpus/dedup/judges/disagreements.csv"
_VERDICTS = "corpus/dedup/paper1_verdicts.csv"
_SAMPLE = "corpus/dedup/audit/human_sample.csv"
_KEY = "corpus/dedup/audit/human_key.csv"

_SAME, _DIFFERENT = "same", "different"  # human labels
_TRUE = "True"  # duplicates.csv same_grade cell


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["draw", "score"])
    parser.add_argument("--duplicates", default=_DUPLICATES)
    args = parser.parse_args()
    if args.command == "draw":
        _draw(args.duplicates)
        return
    _score()


def _draw(duplicates_path: str) -> None:
    rng = random.Random(_SEED)
    removed = _read(duplicates_path)
    conflict = [r for r in removed if r["same_grade"] != _TRUE]
    same = [r for r in removed if r["same_grade"] == _TRUE]

    # Merges: the kept and the removed copy, as the tables see them
    items = [("merged, grade differs", r["doi"], r["kept_ordinal"], r["removed_ordinal"], r["kept_text"],
              r["removed_text"], Verdict.DUPLICATE.value) for r in rng.sample(conflict, _N_GRADE_CONFLICT)]
    items += [("merged, same grade", r["doi"], r["kept_ordinal"], r["removed_ordinal"], r["kept_text"],
               r["removed_text"], Verdict.DUPLICATE.value) for r in rng.sample(same, _N_SAME_GRADE)]

    # Split pairs: the consensus verdict the third judge decided
    final = {(a.doi, a.ordinal_a, a.ordinal_b): a.verdict.value for a in load_adjudications(_VERDICTS)}
    for r in rng.sample(_read(_DISAGREEMENTS), _N_SPLIT):
        key = (r["doi"], int(r["ordinal_a"]), int(r["ordinal_b"]))
        items.append(("judges split", r["doi"], r["ordinal_a"], r["ordinal_b"], r["text_a"], r["text_b"], final[key]))
    rng.shuffle(items)

    with open(_SAMPLE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "doi", "text_a", "text_b", "label", "note"])
        for i, (_, doi, _, _, a, b, _) in enumerate(items, start=1):
            w.writerow([i, doi, a, b, "", ""])
    with open(_KEY, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", "stratum", "doi", "ordinal_a", "ordinal_b", "verdict"])
        for i, (stratum, doi, oa, ob, _, _, verdict) in enumerate(items, start=1):
            w.writerow([i, stratum, doi, oa, ob, verdict])
    print(f"{len(items)} pairs → {_SAMPLE} (label each '{_SAME}' or '{_DIFFERENT}'); key → {_KEY}")


def _score() -> None:
    labels = {r["id"]: r["label"].strip().lower() for r in _read(_SAMPLE)}
    missing = [i for i, label in labels.items() if label not in (_SAME, _DIFFERENT)]
    if missing:
        sys.exit(f"unlabelled or invalid: ids {', '.join(missing)} (use '{_SAME}' or '{_DIFFERENT}')")

    errors_by_stratum: dict[str, list[int]] = {}
    for r in _read(_KEY):
        human = Verdict.DUPLICATE.value if labels[r["id"]] == _SAME else Verdict.DISTINCT.value
        errors_by_stratum.setdefault(r["stratum"], []).append(int(human != r["verdict"]))

    for stratum, errors in sorted(errors_by_stratum.items()):
        print(f"{stratum}: {sum(errors)}/{len(errors)} errors")
    n = sum(len(e) for e in errors_by_stratum.values())
    k = sum(sum(e) for e in errors_by_stratum.values())
    low, high = _clopper_pearson(k, n)
    print(f"all: {k}/{n} errors, {_CONFIDENCE:.0%} CI {low:.1%}–{high:.1%} (stratified sample, not a corpus rate)")


def _clopper_pearson(k: int, n: int) -> tuple[float, float]:
    alpha = 1 - _CONFIDENCE
    low = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    high = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(low), float(high)


def _read(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    main()
