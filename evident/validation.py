"""Score a snapshot against the 12 labelled guidelines and gate it against the published one.

    stored recs {doi: [RawRecommendation]} ─┐
    labelled GT (ACP + ICU xlsx) ───────────┴─▶ score ─▶ SnapshotScore ─┐
                                                                         ├─▶ gate ─▶ GateReport
                                     baseline SnapshotScore (published) ─┘

Both sides are harmonized from their raw columns on read. Pure apart from load_labelled().
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluation.matching import build_similarity_matrix
from evident.domain import (
    AxisStatus,
    Category,
    GateResult,
    GradingFamily,
    HarmonizedGrade,
    RawRecommendation,
)
from evident.harmonization import harmonize
from extraction.datasets import GtMode, load_acp_datasets, load_icu_datasets
from utils.doi import normalize_doi

_GATE_MARGIN = 0.02
_N_BOOTSTRAP = 2000
_CI_PERCENTILES = (2.5, 97.5)
_MATCH_THRESHOLD = 0.65  # BioLORD; the legacy benchmark's value
_N_LABELLED = 12


class ValidationJoinError(ValueError):
    def __init__(self, missing_dois: list[str]):
        super().__init__(f"Labelled guideline(s) without an active run: {', '.join(missing_dois)}")
        self.missing_dois = missing_dois


@dataclass(frozen=True)
class GtRow:
    text: str
    raw_class: str
    raw_loe: str
    category: Category


@dataclass(frozen=True)
class LabelledGuideline:
    doi: str  # normalized
    family: GradingFamily
    gt: list[GtRow]


@dataclass(frozen=True)
class GuidelineScore:
    doi: str
    tp: int
    fp: int
    fn: int
    n_matched: int
    n_strength_ok: int
    n_certainty_ok: int
    n_combined_ok: int
    ungraded_tp: int
    ungraded_fp: int
    ungraded_fn: int


@dataclass(frozen=True)
class SnapshotScore:
    per_guideline: tuple[GuidelineScore, ...]  # sorted by doi


@dataclass(frozen=True)
class VersionMetrics:
    """Micro-pooled over guidelines; None when the denominator is empty."""
    f1: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    strength_accuracy: Optional[float]
    certainty_accuracy: Optional[float]
    combined_accuracy: Optional[float]
    ungraded_precision: Optional[float]
    ungraded_recall: Optional[float]
    f1_ci: tuple[Optional[float], Optional[float]]
    combined_ci: tuple[Optional[float], Optional[float]]


@dataclass(frozen=True)
class GateReport:
    result: GateResult
    n_guidelines: int
    candidate: VersionMetrics
    baseline: Optional[VersionMetrics] = None
    f1_diff: Optional[float] = None
    f1_diff_ci: Optional[tuple[Optional[float], Optional[float]]] = None
    combined_diff: Optional[float] = None
    combined_diff_ci: Optional[tuple[Optional[float], Optional[float]]] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=lambda e: e.value)


def load_labelled() -> list[LabelledGuideline]:
    datasets = load_acp_datasets(GtMode.WITH_UNGRADED) + load_icu_datasets(GtMode.WITH_UNGRADED)
    labelled = [
        LabelledGuideline(
            doi=normalize_doi(ds.doi),
            family=GradingFamily.GRADE,
            gt=[GtRow(str(r["recommendation"]), str(r["raw_class"]), str(r["raw_LOE"]), Category(r["category"]))
                for r in ds.ground_truth_df.to_dict("records")],
        )
        for ds in datasets
    ]
    if len(labelled) != _N_LABELLED:
        raise ValueError(f"Expected {_N_LABELLED} labelled guidelines, found {len(labelled)}")
    return sorted(labelled, key=lambda g: g.doi)


def score(recs_by_doi: dict[str, list[RawRecommendation]], labelled: Sequence[LabelledGuideline],
          similarity_model) -> SnapshotScore:
    recs_by_doi = {normalize_doi(d): recs for d, recs in recs_by_doi.items()}

    # Hard join: a labelled guideline without stored recs would silently inflate scores
    missing = sorted(g.doi for g in labelled if g.doi not in recs_by_doi)
    if missing:
        raise ValidationJoinError(missing)

    scores = [_score_guideline(g, recs_by_doi[g.doi], similarity_model) for g in labelled]
    return SnapshotScore(tuple(sorted(scores, key=lambda s: s.doi)))


def gate(candidate: SnapshotScore, baseline: Optional[SnapshotScore],
         n_boot: int = _N_BOOTSTRAP, seed: int = 0) -> GateReport:
    """Paired guideline-cluster bootstrap; FAIL iff the upper diff bound is below -margin.

    Example: F1 diff CI (-0.10, -0.03) → FAIL; (-0.08, +0.01) → PASS (not a clear regression).
    """
    rng = np.random.default_rng(seed)
    n = len(candidate.per_guideline)
    draws = rng.integers(0, n, size=(n_boot, n))

    cand_counts = _counts(candidate)
    cand_f1, cand_combined = _boot_metrics(cand_counts, draws)
    cand_metrics = _metrics(candidate, cand_f1, cand_combined)

    if baseline is None:
        return GateReport(GateResult.NO_BASELINE, n, cand_metrics)

    if [s.doi for s in baseline.per_guideline] != [s.doi for s in candidate.per_guideline]:
        raise ValueError("Candidate and baseline must score the same guidelines")

    # Same draws for both versions: the pairing removes between-guideline variance
    base_f1, base_combined = _boot_metrics(_counts(baseline), draws)
    base_metrics = _metrics(baseline, base_f1, base_combined)
    f1_diff_ci = _ci(cand_f1 - base_f1)
    combined_diff_ci = _ci(cand_combined - base_combined)

    regressed = any(ci[1] is not None and ci[1] < -_GATE_MARGIN for ci in (f1_diff_ci, combined_diff_ci))
    return GateReport(
        result=GateResult.FAIL if regressed else GateResult.PASS,
        n_guidelines=n,
        candidate=cand_metrics,
        baseline=base_metrics,
        f1_diff=_diff(cand_metrics.f1, base_metrics.f1),
        f1_diff_ci=f1_diff_ci,
        combined_diff=_diff(cand_metrics.combined_accuracy, base_metrics.combined_accuracy),
        combined_diff_ci=combined_diff_ci,
    )


# ── scoring ─────────────────────────────────────────────────────

def _score_guideline(g: LabelledGuideline, recs: list[RawRecommendation], model) -> GuidelineScore:
    extracted = [(r.text, harmonize(r.raw_strength, r.raw_certainty, r.text, r.raw_category, g.family))
                 for r in recs]
    gt = [(row.text, harmonize(row.raw_class, row.raw_loe, row.text, row.category, g.family)) for row in g.gt]

    ext_graded, ext_ungraded = _split(extracted)
    gt_graded, gt_ungraded = _split(gt)

    pairs = _match([t for t, _ in ext_graded], [t for t, _ in gt_graded], model)
    graded = [(ext_graded[i][1], gt_graded[j][1]) for i, j in pairs]
    n_strength = sum(_strength_ok(e, t) for e, t in graded)
    n_certainty = sum(_certainty_ok(e, t) for e, t in graded)
    n_combined = sum(_strength_ok(e, t) and _certainty_ok(e, t) for e, t in graded)

    # Ungraded statements: reported, never gated
    n_ungraded_tp = len(_match([t for t, _ in ext_ungraded], [t for t, _ in gt_ungraded], model))

    return GuidelineScore(
        doi=g.doi,
        tp=len(pairs), fp=len(ext_graded) - len(pairs), fn=len(gt_graded) - len(pairs),
        n_matched=len(pairs), n_strength_ok=n_strength, n_certainty_ok=n_certainty, n_combined_ok=n_combined,
        ungraded_tp=n_ungraded_tp, ungraded_fp=len(ext_ungraded) - n_ungraded_tp,
        ungraded_fn=len(gt_ungraded) - n_ungraded_tp,
    )


def _split(rows: list[tuple[str, HarmonizedGrade]]):
    graded = [r for r in rows if r[1].category == Category.GRADED]
    ungraded = [r for r in rows if r[1].category != Category.GRADED]
    return graded, ungraded


def _match(extracted: list[str], gt: list[str], model) -> list[tuple[int, int]]:
    """Hungarian matching above the threshold, as evaluation.match_recommendations, but keeping indices."""
    if not extracted or not gt:
        return []
    sim = build_similarity_matrix(extracted, gt, model)
    rows, cols = linear_sum_assignment(1.0 - sim)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if sim[r, c] >= _MATCH_THRESHOLD]


def _strength_ok(e: HarmonizedGrade, t: HarmonizedGrade) -> bool:
    both_mapped = e.strength_status == t.strength_status == AxisStatus.MAPPED
    return both_mapped and (e.strength, e.direction) == (t.strength, t.direction)


def _certainty_ok(e: HarmonizedGrade, t: HarmonizedGrade) -> bool:
    both_mapped = e.certainty_status == t.certainty_status == AxisStatus.MAPPED
    return both_mapped and e.certainty == t.certainty


# ── metrics and bootstrap ───────────────────────────────────────

def _ratio(num: float, den: float) -> Optional[float]:
    return num / den if den else None


def _diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or b is None else a - b


def _sum(s: SnapshotScore, name: str) -> int:
    return sum(getattr(g, name) for g in s.per_guideline)


def _metrics(s: SnapshotScore, boot_f1: np.ndarray, boot_combined: np.ndarray) -> VersionMetrics:
    tp, fp, fn, matched = (_sum(s, n) for n in ("tp", "fp", "fn", "n_matched"))
    u_tp, u_fp, u_fn = (_sum(s, n) for n in ("ungraded_tp", "ungraded_fp", "ungraded_fn"))
    return VersionMetrics(
        f1=_ratio(2 * tp, 2 * tp + fp + fn),
        precision=_ratio(tp, tp + fp),
        recall=_ratio(tp, tp + fn),
        strength_accuracy=_ratio(_sum(s, "n_strength_ok"), matched),
        certainty_accuracy=_ratio(_sum(s, "n_certainty_ok"), matched),
        combined_accuracy=_ratio(_sum(s, "n_combined_ok"), matched),
        ungraded_precision=_ratio(u_tp, u_tp + u_fp),
        ungraded_recall=_ratio(u_tp, u_tp + u_fn),
        f1_ci=_ci(boot_f1),
        combined_ci=_ci(boot_combined),
    )


def _counts(s: SnapshotScore) -> np.ndarray:
    """(n_guidelines, 5): tp, fp, fn, matched, combined_ok."""
    return np.array([[g.tp, g.fp, g.fn, g.n_matched, g.n_combined_ok] for g in s.per_guideline], dtype=float)


def _boot_metrics(counts: np.ndarray, draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Micro F1 and combined accuracy per bootstrap draw; NaN where the denominator is 0."""
    tp, fp, fn, matched, combined = (counts[draws].sum(axis=1)[:, k] for k in range(5))
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
        accuracy = np.where(matched > 0, combined / matched, np.nan)
    return f1, accuracy


def _ci(values: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return None, None
    low, high = np.percentile(finite, _CI_PERCENTILES)
    return float(low), float(high)
