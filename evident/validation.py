"""Score a snapshot against the 12 labelled guidelines and gate it against the published one.

    stored recs {doi: [RawRecommendation]} ─┐
    labelled GT (ACP + ICU xlsx) ───────────┴─▶ score ─▶ SnapshotScore ─┐
                                                                         ├─▶ gate ─▶ GateReport
                                     baseline SnapshotScore (published) ─┘

    held-out GT (separate dir) ─▶ load_heldout ─▶ score ─▶ heldout_report  (reported, never gated)
    M3 spot-check JSON ─────────▶ load_spotcheck                          (exploratory reference lists)

Both sides are harmonized from their raw columns on read. Pure apart from the load_* functions.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from evaluation.matching import build_similarity_matrix
from evident.domain import (
    AxisStatus,
    Category,
    Certainty,
    Direction,
    GateResult,
    GradingFamily,
    HarmonizedGrade,
    RawRecommendation,
    Strength,
)
from evident.harmonization import harmonize
from extraction.datasets import GtMode, load_acp_datasets, load_icu_datasets
from utils.doi import normalize_doi

_GATE_MARGIN = 0.02
_N_BOOTSTRAP = 2000
_CI_PERCENTILES = (2.5, 97.5)
_MATCH_THRESHOLD = 0.65  # BioLORD; the legacy benchmark's value
_N_LABELLED = 12
_WILSON_Z = 1.959963984540054  # two-sided 95%

_XLSX_SUFFIX = ".xlsx"
_EXCEL_LOCK_PREFIX = "~$"
_HELDOUT_SHEET = "labels"
_HELDOUT_COLUMNS = ("Key", "Title", "DOI", "recommendation", "class", "LOE", "page")
_FIRST_EXCEL_DATA_LINE = 2  # line 1 is the header
_HELDOUT_INTERVAL_NOTE = "Wilson 95% interval on recommendations; ignores clustering within guidelines"

_SPOTCHECK_SUFFIX = ".spotcheck.json"
_REFERENCE_STRENGTHS = {"strong": Strength.STRONG, "weak": Strength.WEAK}


class ValidationJoinError(ValueError):
    def __init__(self, missing_dois: list[str]):
        super().__init__(f"Labelled guideline(s) without an active run: {', '.join(missing_dois)}")
        self.missing_dois = missing_dois


class HeldoutLabelError(ValueError):
    """All problems found in a held-out label set; the package refuses to render without a clean set."""

    def __init__(self, problems: list[str]):
        super().__init__("Invalid held-out labels:\n" + "\n".join(problems))
        self.problems = problems


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


@dataclass(frozen=True)
class HeldoutReport:
    """Held-out guidelines scored like the gate, micro-pooled; no bootstrap (too few clusters)."""
    n_guidelines: int
    per_guideline: tuple[GuidelineScore, ...]
    tp: int
    fp: int
    fn: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    precision_ci: tuple[Optional[float], Optional[float]]
    recall_ci: tuple[Optional[float], Optional[float]]
    n_matched: int
    strength_accuracy: Optional[float]   # strength with direction, matched graded pairs
    certainty_accuracy: Optional[float]
    combined_accuracy: Optional[float]
    strength_ci: tuple[Optional[float], Optional[float]]
    certainty_ci: tuple[Optional[float], Optional[float]]
    combined_ci: tuple[Optional[float], Optional[float]]
    ungraded_precision: Optional[float]  # exploratory: v2 does not target ungraded statements
    ungraded_recall: Optional[float]
    interval_note: str = _HELDOUT_INTERVAL_NOTE


@dataclass(frozen=True)
class SpotcheckGuideline:
    """One M3 spot-check guideline: which extracted rows matched, and the reviewer's reference list."""
    doi: str
    matched_ordinals: tuple[int, ...]
    reference: tuple[HarmonizedGrade, ...]


def load_labelled() -> list[LabelledGuideline]:
    datasets = load_acp_datasets(GtMode.WITH_UNGRADED) + load_icu_datasets(GtMode.WITH_UNGRADED)
    labelled = _to_labelled(datasets)
    if len(labelled) != _N_LABELLED:
        raise ValueError(f"Expected {_N_LABELLED} labelled guidelines, found {len(labelled)}")
    return labelled


def load_heldout(directory: str, expected_dois: Sequence[str], labelled_dois: set[str]) -> list[LabelledGuideline]:
    """Exactly the prespecified guidelines, each complete and traceable; never the development labels.

    Fails closed: any problem raises HeldoutLabelError listing all of them.
    """
    problems = _heldout_problems(directory, {normalize_doi(d) for d in expected_dois},
                                 {normalize_doi(d) for d in labelled_dois})
    if problems:
        raise HeldoutLabelError(problems)
    return _to_labelled(load_icu_datasets(GtMode.WITH_UNGRADED, directory))


def heldout_report(s: SnapshotScore) -> HeldoutReport:
    tp, fp, fn, matched = (_sum(s, n) for n in ("tp", "fp", "fn", "n_matched"))
    strength_ok, certainty_ok, combined_ok = (_sum(s, n) for n in ("n_strength_ok", "n_certainty_ok",
                                                                    "n_combined_ok"))
    u_tp, u_fp, u_fn = (_sum(s, n) for n in ("ungraded_tp", "ungraded_fp", "ungraded_fn"))
    return HeldoutReport(
        n_guidelines=len(s.per_guideline),
        per_guideline=s.per_guideline,
        tp=tp, fp=fp, fn=fn,
        precision=_ratio(tp, tp + fp),
        recall=_ratio(tp, tp + fn),
        f1=_ratio(2 * tp, 2 * tp + fp + fn),
        precision_ci=_wilson(tp, tp + fp),
        recall_ci=_wilson(tp, tp + fn),
        n_matched=matched,
        strength_accuracy=_ratio(strength_ok, matched),
        certainty_accuracy=_ratio(certainty_ok, matched),
        combined_accuracy=_ratio(combined_ok, matched),
        strength_ci=_wilson(strength_ok, matched),
        certainty_ci=_wilson(certainty_ok, matched),
        combined_ci=_wilson(combined_ok, matched),
        ungraded_precision=_ratio(u_tp, u_tp + u_fp),
        ungraded_recall=_ratio(u_tp, u_tp + u_fn),
    )


def load_spotcheck(directory: str) -> list[SpotcheckGuideline]:
    """Reviewer reference lists of the M3 spot check (single reviewer; exploratory), sorted by DOI."""
    out = []
    for name in sorted(f for f in os.listdir(directory) if f.endswith(_SPOTCHECK_SUFFIX)):
        with open(os.path.join(directory, name), encoding="utf-8") as f:
            data = json.load(f)
        out.append(SpotcheckGuideline(
            doi=normalize_doi(data["doi"]),
            matched_ordinals=tuple(sorted(m["extracted_id"] for m in data["matches"])),
            reference=tuple(_reference_grade(r) for r in data["reference"]),
        ))
    return sorted(out, key=lambda g: g.doi)


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


# ── loading ─────────────────────────────────────────────────────

def _to_labelled(datasets) -> list[LabelledGuideline]:
    labelled = [
        LabelledGuideline(
            doi=normalize_doi(ds.doi),
            family=GradingFamily.GRADE,
            gt=[GtRow(str(r["recommendation"]), str(r["raw_class"]), str(r["raw_LOE"]), Category(r["category"]))
                for r in ds.ground_truth_df.to_dict("records")],
        )
        for ds in datasets
    ]
    return sorted(labelled, key=lambda g: g.doi)


def _heldout_problems(directory: str, expected: set[str], labelled: set[str]) -> list[str]:
    if not os.path.isdir(directory):
        return [f"{directory}: not a directory"]

    problems = [f"{doi}: also a development label; held-out labels must be new" for doi in sorted(expected & labelled)]
    names = sorted(f for f in os.listdir(directory)
                   if f.endswith(_XLSX_SUFFIX) and not f.startswith(_EXCEL_LOCK_PREFIX))

    found: dict[str, str] = {}  # doi → file
    for name in names:
        doi = _heldout_file_problems(os.path.join(directory, name), name, problems)
        if doi is None:
            continue
        if doi in found:
            problems.append(f"{name}: {doi} already labelled in {found[doi]}")
            continue
        found[doi] = name

    problems += [f"{doi}: no label file" for doi in sorted(expected - set(found))]
    problems += [f"{found[doi]}: {doi} is not a prespecified held-out guideline"
                 for doi in sorted(set(found) - expected)]
    return problems


def _heldout_file_problems(path: str, name: str, problems: list[str]) -> Optional[str]:
    """Check one label file; append its problems; return its normalized DOI, or None if unreadable."""
    sheets = pd.ExcelFile(path).sheet_names
    if not sheets or sheets[0] != _HELDOUT_SHEET:
        problems.append(f"{name}: first sheet must be '{_HELDOUT_SHEET}' (the loader reads the first sheet)")
        return None

    df = pd.read_excel(path, sheet_name=_HELDOUT_SHEET)
    missing = [c for c in _HELDOUT_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"{name}: missing column(s) {', '.join(missing)}")
        return None

    dois = {normalize_doi(d) for d in df["DOI"].dropna().astype(str) if d.strip()}
    if len(dois) != 1:
        problems.append(f"{name}: expected one DOI, found {sorted(dois) or 'none'}")
        return None
    doi = dois.pop()

    labelled = df[df["recommendation"].map(_filled)]
    if labelled.empty:
        problems.append(f"{name}: no labelled statement")

    seen: dict[str, int] = {}
    for index, row in labelled.iterrows():
        line = index + _FIRST_EXCEL_DATA_LINE
        if not _filled(row["class"]):
            problems.append(f"{name} line {line}: empty class")
        elif not _filled(row["LOE"]) and _is_graded(row):
            problems.append(f"{name} line {line}: empty LOE")  # 'NA' is allowed for ungraded statements only
        if not _is_page(row["page"]):
            problems.append(f"{name} line {line}: page '{row['page']}' is not a page number")

        text = " ".join(str(row["recommendation"]).lower().split())
        if text in seen:
            problems.append(f"{name} line {line}: duplicate of line {seen[text]}")
            continue
        seen[text] = line
    return doi


def _filled(value) -> bool:
    return not pd.isna(value) and bool(str(value).strip())


def _is_graded(row) -> bool:
    grade = harmonize(str(row["class"]), "", str(row["recommendation"]), Category.GRADED, GradingFamily.GRADE)
    return grade.category == Category.GRADED


def _is_page(value) -> bool:
    """A positive whole number; Excel hands integers back as floats (3.0)."""
    if pd.isna(value):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number >= 1 and number.is_integer()


def _reference_grade(r: dict) -> HarmonizedGrade:
    """Spot-check reference row → grade; 'mixed: …' certainty or a missing direction stay unmapped."""
    category = Category(r["category"])
    if category != Category.GRADED:
        return HarmonizedGrade(category, None, None, None, AxisStatus.UNGRADED, AxisStatus.UNGRADED)

    strength = _REFERENCE_STRENGTHS.get(r.get("strength") or "")
    direction = next((d for d in Direction if d.value == r.get("direction")), None)
    certainty = next((c for c in Certainty if c.value == r.get("certainty")), None)
    return HarmonizedGrade(
        category=category,
        strength=strength,
        direction=direction if strength else None,
        certainty=certainty,
        strength_status=AxisStatus.MAPPED if strength else AxisStatus.UNMAPPED,
        certainty_status=AxisStatus.MAPPED if certainty else AxisStatus.UNMAPPED,
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


def _wilson(successes: int, n: int) -> tuple[Optional[float], Optional[float]]:
    """Wilson score interval, e.g. 9/10 → (0.596, 0.982); empty → (None, None)."""
    if not n:
        return None, None
    p = successes / n
    z2 = _WILSON_Z ** 2
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = _WILSON_Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return centre - half, centre + half


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
