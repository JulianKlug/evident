"""Paper analysis package: pure tables over one frozen snapshot's harmonized rows.

    [HarmonizedRow] ─┬─ current, graded ─▶ table1 (pooled %, guideline-cluster CI) · table2 · guideline_mean (S1)
                     ├─ all editions ────▶ trend points (figure data) · raw_mapping
                     └─ + spot-check refs ▶ spotcheck_bias (exploratory)
    [Candidate] + [ManifestEntry] ──────▶ corpus_flow · excluded

Unit of analysis: one extracted recommendation. A % uses only graded rows whose axis is
MAPPED (analytics' denominator rule). Values stay unrounded; publishing rounds.
Empty denominator → None, rendered "NA", never 0.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np

from evident.analytics import Edition, HarmonizedRow
from evident.corpus import Candidate, CandidateDecision, ExclusionReason, ManifestEntry
from evident.dedup import RemovedRow
from evident.domain import AxisStatus, Category, Certainty, Direction, ExtractorVersion, HarmonizedGrade, Strength
from evident.validation import HeldoutReport, SpotcheckGuideline

ALL = "ALL"
EDITION_RULE = ("current = the latest edition of each topic_id that no manifest row supersedes "
                "(supersedes_doi); superseded editions count in trends only")

_PCT = 100.0
_CI_PERCENTILES = (2.5, 97.5)
_HIGH_MODERATE = {Certainty.HIGH, Certainty.MODERATE}
_CERTAINTY_ORDER = (Certainty.HIGH, Certainty.MODERATE, Certainty.LOW, Certainty.VERY_LOW)
_STRENGTH_ORDER = (Strength.STRONG, Strength.WEAK)
_CERTAINTY_TABLE = "certainty"


class Metric(str, Enum):
    STRONG = "strong"                  # strong for or against / strength mapped
    AGAINST = "against"                # against, strong or conditional / strength mapped
    HIGH_MODERATE = "high_moderate"    # high or moderate / certainty mapped
    UNGRADED_SHARE = "ungraded_share"  # ungraded / all rows; spot-check diagnostic only, never a finding


PRIMARY_METRICS = (Metric.STRONG, Metric.AGAINST, Metric.HIGH_MODERATE)


# ── inputs ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class AnalysisConfig:
    """What one paper's package is computed from and checked against (e.g. configs/paper1.json)."""
    name: str
    snapshot_id: int
    search_start: str
    search_cutoff: str
    db_sha256: str
    manifest_sha256: str
    candidates_sha256: str
    candidates_path: str
    n_guidelines: int
    n_current: int
    bootstrap_n: int
    bootstrap_seed: int
    heldout_dois: tuple[str, ...]
    spotcheck_dir: str
    focused_update_overlap_max: int  # estimated graded recs counted twice by focused updates
    heldout_labels_sha256: dict[str, str]  # {label file: SHA-256}, frozen with the labels
    dedup_verdicts_path: Optional[str] = None    # frozen duplicate verdicts (spec §5); None → no dedup
    dedup_verdicts_sha256: Optional[str] = None

    @classmethod
    def from_json(cls, text: str) -> "AnalysisConfig":
        data = json.loads(text)
        data["heldout_dois"] = tuple(data["heldout_dois"])
        return cls(**data)


# ── outputs ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class Proportion:
    num: int
    den: int
    pct: Optional[float]               # None when den == 0
    ci_low: Optional[float] = None     # guideline-cluster bootstrap, percent
    ci_high: Optional[float] = None
    undefined_draws: int = 0           # bootstrap draws with an empty denominator, dropped


@dataclass(frozen=True)
class Table1Row:
    society: str
    n_guidelines: int
    n_recs: int
    n_graded: int
    strong: Proportion
    against: Proportion
    high_moderate: Proportion
    n_strength_unmapped: int
    n_certainty_unmapped: int
    n_certainty_ungraded: int


@dataclass(frozen=True)
class Table2Row:
    society: str
    table: str        # "certainty", or the strength of a strength × certainty row
    certainty: Certainty
    n: int
    den: int          # certainty table: certainty mapped; crosstab: that strength, both axes mapped
    pct: Optional[float]


@dataclass(frozen=True)
class GuidelineMeanRow:
    """Table S1 (design Decision 10): each guideline weighs the same, against the pooled estimate."""
    society: str
    metric: Metric
    n_guidelines: int          # contributing: denominator > 0 on this axis
    n_guidelines_omitted: int  # zero denominator on this axis
    mean_pct: Optional[float]
    pooled_pct: Optional[float]
    diff_pp: Optional[float]   # mean − pooled, percentage points


@dataclass(frozen=True)
class TrendPoint:
    society: str
    year: int
    metric: Metric
    n_guidelines: int
    n_graded: int
    num: int
    den: int
    pct: Optional[float]


@dataclass(frozen=True)
class FigurePoint:
    """One plotted value of fig1–fig4; enough to redraw every figure."""
    figure: str
    metric: str
    society: str
    year: Optional[int]
    strength: str
    certainty: str
    n_guidelines: Optional[int]
    num: int
    den: int
    pct: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]


@dataclass(frozen=True)
class FlowRow:
    stage: str  # candidates, source, excluded, included, current, superseded
    item: str   # "all", a source, an exclusion reason or a society
    n: int


@dataclass(frozen=True)
class ExcludedRow:
    doi: str
    society: str
    year: Optional[int]
    title: str
    reason: str
    duplicate_of: str
    source: str


@dataclass(frozen=True)
class MappingRow:
    raw_strength: str
    raw_certainty: str
    category: Category
    strength: Optional[Strength]
    direction: Optional[Direction]
    certainty: Optional[Certainty]
    strength_status: AxisStatus
    certainty_status: AxisStatus
    n_all_editions: int
    n_current: int


@dataclass(frozen=True)
class BiasRow:
    """M3 spot check: does the published headline move against the reviewer's reference list?"""
    metric: Metric
    n_guidelines: int
    published: Proportion
    correct: Optional[Proportion]  # matched rows only; None for the ungraded share
    truth: Proportion
    diff_pp: Optional[float]       # published − truth
    diff_ci_low: Optional[float]
    diff_ci_high: Optional[float]


@dataclass(frozen=True)
class PaperPackage:
    """Everything `publish --package` renders besides the standard outputs."""
    config: AnalysisConfig
    table1: list[Table1Row]
    table2: list[Table2Row]
    guideline_mean: list[GuidelineMeanRow]
    figure_data: list[FigurePoint]
    corpus_flow: list[FlowRow]
    excluded: list[ExcludedRow]
    raw_mapping: list[MappingRow]
    spotcheck_bias: list[BiasRow]
    validation_development: dict  # gate report of the 12 labelled guidelines (in-sample)
    validation_heldout: HeldoutReport
    heldout_societies: tuple[str, ...]
    methods_facts: dict
    provenance: dict              # sources, versions and checksums for analysis_manifest.json
    duplicates: list[RemovedRow] = field(default_factory=list)  # within-guideline copies removed
    draft_template: Optional[str] = None  # manuscript template (private); None → no draft.md
    table1_without_dedup: list[Table1Row] = field(default_factory=list)  # sensitivity: copies kept


# ── primary tables ──────────────────────────────────────────────

def table1(rows: Sequence[HarmonizedRow], n_boot: int, seed: int) -> list[Table1Row]:
    """Per society and ALL, current editions: pooled per recommendation, CI from guideline clusters.

    ALL pools recommendations (SCCM-dominated), not the mean of society percentages.
    """
    current = _current(rows)
    by_doi = _grades_by_doi(current)
    society_of = {r.doi: r.society for r in current}
    cis = {m: cluster_ci([(society_of[d], *_counts(g, m)) for d, g in by_doi.items()], n_boot, seed)
           for m in PRIMARY_METRICS}

    out = []
    for society, group in _groups(current):
        grades = [r.grade for r in group]
        graded = _graded(grades)
        p = {m: _proportion(*_counts(grades, m), cis[m].get(society)) for m in PRIMARY_METRICS}
        out.append(Table1Row(
            society=society,
            n_guidelines=len({r.doi for r in group}),
            n_recs=len(grades),
            n_graded=len(graded),
            strong=p[Metric.STRONG], against=p[Metric.AGAINST], high_moderate=p[Metric.HIGH_MODERATE],
            n_strength_unmapped=sum(g.strength_status == AxisStatus.UNMAPPED for g in graded),
            n_certainty_unmapped=sum(g.certainty_status == AxisStatus.UNMAPPED for g in graded),
            n_certainty_ungraded=sum(g.certainty_status == AxisStatus.UNGRADED for g in graded),
        ))
    return out


def table2(rows: Sequence[HarmonizedRow]) -> list[Table2Row]:
    """Certainty distribution, then strength × certainty, per society and ALL (current editions).

    Same counts as analytics.certainty_distribution / strength_by_certainty, unrounded.
    """
    out = []
    for society, group in _groups(_current(rows)):
        graded = _graded([r.grade for r in group])
        certainty_mapped = [g for g in graded if g.certainty_status == AxisStatus.MAPPED]
        for level in _CERTAINTY_ORDER:
            n = sum(g.certainty == level for g in certainty_mapped)
            out.append(Table2Row(society, _CERTAINTY_TABLE, level, n, len(certainty_mapped),
                                 _pct(n, len(certainty_mapped))))

        both = [g for g in certainty_mapped if g.strength_status == AxisStatus.MAPPED]
        for strength in _STRENGTH_ORDER:
            of_strength = [g for g in both if g.strength == strength]
            for level in _CERTAINTY_ORDER:
                n = sum(g.certainty == level for g in of_strength)
                out.append(Table2Row(society, strength.value, level, n, len(of_strength),
                                     _pct(n, len(of_strength))))
    return out


def guideline_mean(rows: Sequence[HarmonizedRow]) -> list[GuidelineMeanRow]:
    """Unweighted mean of each guideline's own unrounded %; zero-denominator guidelines left out and counted.

    Example: guideline A 1/1 strong, B 1/9 → mean 55.6%, pooled 2/10 = 20.0%, diff +35.6 pp.
    """
    out = []
    for society, group in _groups(_current(rows)):
        by_doi = _grades_by_doi(group)
        grades = [r.grade for r in group]
        for metric in PRIMARY_METRICS:
            ratios = [_ratio(*_counts(g, metric)) for g in by_doi.values()]
            defined = [r for r in ratios if r is not None]
            mean = _PCT * sum(defined) / len(defined) if defined else None
            pooled = _pct(*_counts(grades, metric))
            out.append(GuidelineMeanRow(
                society=society, metric=metric,
                n_guidelines=len(defined), n_guidelines_omitted=len(ratios) - len(defined),
                mean_pct=mean, pooled_pct=pooled,
                diff_pp=None if mean is None or pooled is None else mean - pooled,
            ))
    return out


def trend_points(rows: Sequence[HarmonizedRow]) -> list[TrendPoint]:
    """Per society and ALL × first-online year, all editions; % strong and % high/moderate."""
    out = []
    for society, group in _groups(list(rows)):
        for year in sorted({r.year for r in group}):
            in_year = [r for r in group if r.year == year]
            grades = [r.grade for r in in_year]
            for metric in (Metric.STRONG, Metric.HIGH_MODERATE):
                num, den = _counts(grades, metric)
                out.append(TrendPoint(society, year, metric, len({r.doi for r in in_year}),
                                      len(_graded(grades)), num, den, _pct(num, den)))
    return out


def figure_data(t1: Sequence[Table1Row], t2: Sequence[Table2Row],
                trends: Sequence[TrendPoint]) -> list[FigurePoint]:
    """Every point behind fig1 (certainty), fig2 (strong), fig3 (strength × certainty), fig4 (trends)."""
    out = []
    for r in t2:
        figure = "fig1_certainty_by_society" if r.table == _CERTAINTY_TABLE else "fig3_strength_by_certainty"
        strength = "" if r.table == _CERTAINTY_TABLE else r.table
        out.append(FigurePoint(figure, "certainty_share", r.society, None, strength, r.certainty.value, None,
                               r.n, r.den, r.pct, None, None))
    for r in t1:
        out.append(FigurePoint("fig2_strong_by_society", Metric.STRONG.value, r.society, None, "", "",
                               r.n_guidelines, r.strong.num, r.strong.den, r.strong.pct,
                               r.strong.ci_low, r.strong.ci_high))
    for p in trends:
        out.append(FigurePoint("fig4_trends", p.metric.value, p.society, p.year, "", "", p.n_guidelines,
                               p.num, p.den, p.pct, None, None))
    return out


# ── uncertainty ─────────────────────────────────────────────────

@dataclass(frozen=True)
class ClusterCi:
    low: Optional[float]   # percent
    high: Optional[float]
    undefined_draws: int


def cluster_ci(clusters: Sequence[tuple[str, int, int]], n_boot: int, seed: int) -> dict[str, ClusterCi]:
    """Guideline-cluster percentile bootstrap of a pooled ratio, per society and ALL.

    clusters: one (society, numerator, denominator) per guideline. Each draw resamples guidelines
    with replacement within each society; ALL pools that draw's society resamples (stratified), so
    the society mix is fixed. Draws with an empty denominator are dropped and counted.

    Example: SCCM [(SCCM, 1, 2), (SCCM, 3, 4)] → each draw is 2 guidelines from those two.
    """
    societies = sorted({s for s, _, _ in clusters})
    rng = np.random.default_rng(seed)
    all_num = np.zeros(n_boot)
    all_den = np.zeros(n_boot)

    out = {}
    for society in societies:
        counts = np.array([(n, d) for s, n, d in clusters if s == society], dtype=float)
        draws = rng.integers(0, len(counts), size=(n_boot, len(counts)))
        num = counts[draws, 0].sum(axis=1)
        den = counts[draws, 1].sum(axis=1)
        all_num += num
        all_den += den
        out[society] = _percentile_ci(num, den)

    out[ALL] = _percentile_ci(all_num, all_den) if societies else ClusterCi(None, None, 0)
    return out


def _percentile_ci(num: np.ndarray, den: np.ndarray) -> ClusterCi:
    defined = den > 0
    if not defined.any():
        return ClusterCi(None, None, int(den.size))
    low, high = np.percentile(_PCT * num[defined] / den[defined], _CI_PERCENTILES)
    return ClusterCi(float(low), float(high), int((~defined).sum()))


# ── corpus ──────────────────────────────────────────────────────

def corpus_flow(candidates: Sequence[Candidate], entries: Sequence[ManifestEntry],
                current_dois: set[str]) -> list[FlowRow]:
    """Search → screening → inclusion counts; every exclusion reason listed, zero counts included."""
    included = [c for c in candidates if c.decision == CandidateDecision.INCLUDE]
    reasons = Counter(c.reason for c in candidates if c.decision == CandidateDecision.EXCLUDE)
    sources = Counter(c.source for c in candidates)
    current = [e for e in entries if e.doi in current_dois]

    out = [FlowRow("candidates", "all", len(candidates))]
    out += [FlowRow("source", s, sources[s]) for s in sorted(sources)]
    out += [FlowRow("excluded", r.value, reasons[r]) for r in ExclusionReason]
    out.append(FlowRow("included", "all", len(included)))
    out += _by_society_counts("included", entries)
    out.append(FlowRow("current", "all", len(current)))
    out += _by_society_counts("current", current)
    out.append(FlowRow("superseded", "all", len(entries) - len(current)))
    return out


def excluded_guidelines(candidates: Sequence[Candidate]) -> list[ExcludedRow]:
    """Every excluded candidate with its one reason, by reason then DOI."""
    excluded = [c for c in candidates if c.decision == CandidateDecision.EXCLUDE]
    return [ExcludedRow(c.doi, c.society, c.year, c.title, c.reason.value, c.duplicate_of or "", c.source)
            for c in sorted(excluded, key=lambda c: (c.reason.value, c.doi))]


def raw_mapping(rows: Sequence[HarmonizedRow]) -> list[MappingRow]:
    """Every observed raw grade → harmonized grade, with row counts; most frequent first."""
    all_editions = Counter(_mapping_key(r) for r in rows)
    current = Counter(_mapping_key(r) for r in rows if r.edition == Edition.CURRENT)
    keys = sorted(all_editions, key=lambda k: (-all_editions[k], tuple(str(v) for v in k)))
    return [MappingRow(*k, all_editions[k], current[k]) for k in keys]


def _mapping_key(r: HarmonizedRow) -> tuple:
    g = r.grade
    return (r.raw_strength, r.raw_certainty, g.category, g.strength, g.direction, g.certainty,
            g.strength_status, g.certainty_status)


def _by_society_counts(stage: str, entries: Sequence[ManifestEntry]) -> list[FlowRow]:
    counts = Counter(e.society for e in entries)
    return [FlowRow(stage, s, counts[s]) for s in sorted(counts)]


def methods_facts(config: AnalysisConfig, entries: Sequence[ManifestEntry], current_dois: set[str],
                  candidates: Sequence[Candidate], rows: Sequence[HarmonizedRow],
                  queries: Sequence[tuple[str, str, str]], version: ExtractorVersion,
                  n_development: int, dedup: Optional[dict] = None) -> dict:
    """Counts and provenance the Methods section states; queries are (society, source, term).

    dedup: the duplicate step's counts (pipeline), or None when the config names no verdicts.
    """
    current = [e for e in entries if e.doi in current_dois]
    current_by_topic = {e.topic_id: e.doi for e in current}
    current_rows = _current(rows)
    years = [e.year for e in entries]
    return {
        "search": {
            "start": config.search_start,
            "cutoff": config.search_cutoff,
            "cutoff_year": int(config.search_cutoff[:4]),  # a partial calendar year
            "queries": [{"society": s, "source": src, "term": t} for s, src, t in queries],
            "n_candidates": len(candidates),
        },
        "corpus": {
            "n_included": len(entries),
            "n_current": len(current),
            "n_superseded": len(entries) - len(current),
            "included_by_society": dict(sorted(Counter(e.society for e in entries).items())),
            "current_by_society": dict(sorted(Counter(e.society for e in current).items())),
            "year_min": min(years) if years else None,
            "year_max": max(years) if years else None,
            "superseded": [{"doi": e.doi, "topic_id": e.topic_id, "year": e.year,
                            "current_edition": current_by_topic.get(e.topic_id, "")}
                           for e in sorted(entries, key=lambda e: e.doi) if e.doi not in current_dois],
            "joint": [{"doi": e.doi, "society": e.society, "joint_with": list(e.joint_with)}
                      for e in sorted(entries, key=lambda e: e.doi) if e.joint_with],
            "edition_rule": EDITION_RULE,
            "focused_update_overlap_max": config.focused_update_overlap_max,
        },
        "recommendations": {
            "n_current": len(current_rows),
            "n_graded_current": len(_graded([r.grade for r in current_rows])),
            "n_all_editions": len(rows),
            "n_graded_all_editions": len(_graded([r.grade for r in rows])),
        },
        "extractor": {
            "version_id": version.id,
            "model_name": version.model_name,
            "model_digest": version.model_digest,
            "embedding_revision": version.embedding_revision,
            "classifier_sha256": version.classifier_sha256,
            "config": json.loads(version.config_json),
        },
        "validation": {
            "n_development_guidelines": n_development,
            "heldout_dois": list(config.heldout_dois),
            "n_heldout_guidelines": len(config.heldout_dois),
        },
        "bootstrap": {"n": config.bootstrap_n, "seed": config.bootstrap_seed,
                      "percentiles": list(_CI_PERCENTILES)},
        "dedup": {"applied": False} if dedup is None else {"applied": True, **dedup},
    }


# ── M3 spot check (exploratory) ─────────────────────────────────

def spotcheck_bias(rows: Sequence[HarmonizedRow], spot: Sequence[SpotcheckGuideline],
                   n_boot: int, seed: int) -> list[BiasRow]:
    """Published (all extracted rows) vs correct (rows matched to a reference) vs truth (reference list).

    Same metric definitions as the headline. CI: unstratified guideline-cluster bootstrap of the
    published − truth difference. Raises ValueError for a spot-check DOI or matched row without extracted rows.
    """
    by_doi: dict[str, list[HarmonizedRow]] = {}
    for r in rows:
        by_doi.setdefault(r.doi, []).append(r)
    missing = sorted(g.doi for g in spot if g.doi not in by_doi)
    if missing:
        raise ValueError(f"Spot-check guideline(s) without extracted rows: {', '.join(missing)}")
    # The JSONs name rows by ordinal of the reviewed run; one that the stored run lacks means another run
    ordinals = {g.doi: {r.ordinal for r in by_doi[g.doi]} for g in spot}
    unknown = [f"{g.doi} row {o}" for g in spot for o in g.matched_ordinals if o not in ordinals[g.doi]]
    if unknown:
        raise ValueError(f"Spot-check rows missing from the extracted run: {', '.join(unknown)}")

    published = {g.doi: [r.grade for r in by_doi[g.doi]] for g in spot}
    correct = {g.doi: [r.grade for r in by_doi[g.doi] if r.ordinal in set(g.matched_ordinals)] for g in spot}
    truth = {g.doi: list(g.reference) for g in spot}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(spot), size=(n_boot, len(spot))) if spot else None

    out = []
    for metric in PRIMARY_METRICS + (Metric.UNGRADED_SHARE,):
        pub = _pooled(published, metric)
        tru = _pooled(truth, metric)
        diff = None if pub.pct is None or tru.pct is None else pub.pct - tru.pct
        low, high = _diff_ci(published, truth, metric, draws)
        out.append(BiasRow(
            metric=metric, n_guidelines=len(spot), published=pub,
            correct=None if metric == Metric.UNGRADED_SHARE else _pooled(correct, metric),
            truth=tru, diff_pp=diff, diff_ci_low=low, diff_ci_high=high,
        ))
    return out


def _pooled(grades_by_doi: dict[str, list[HarmonizedGrade]], metric: Metric) -> Proportion:
    return _proportion(*_counts([g for grades in grades_by_doi.values() for g in grades], metric))


def _diff_ci(published: dict[str, list[HarmonizedGrade]], truth: dict[str, list[HarmonizedGrade]],
             metric: Metric, draws: Optional[np.ndarray]) -> tuple[Optional[float], Optional[float]]:
    if draws is None:
        return None, None
    dois = list(published)
    pub = np.array([_counts(published[d], metric) for d in dois], dtype=float)
    tru = np.array([_counts(truth[d], metric) for d in dois], dtype=float)
    pub_num, pub_den = pub[draws, 0].sum(axis=1), pub[draws, 1].sum(axis=1)
    tru_num, tru_den = tru[draws, 0].sum(axis=1), tru[draws, 1].sum(axis=1)

    defined = (pub_den > 0) & (tru_den > 0)
    if not defined.any():
        return None, None
    diff = _PCT * (pub_num[defined] / pub_den[defined] - tru_num[defined] / tru_den[defined])
    low, high = np.percentile(diff, _CI_PERCENTILES)
    return float(low), float(high)


# ── metric definitions ──────────────────────────────────────────

def _counts(grades: Sequence[HarmonizedGrade], metric: Metric) -> tuple[int, int]:
    """(numerator, denominator) of one metric over a set of rows.

    Strong and against share the strength-mapped denominator; a no-direction statement is
    no_recommendation (not graded), so it is never counted as against.
    """
    if metric == Metric.UNGRADED_SHARE:
        return len(grades) - len(_graded(grades)), len(grades)

    graded = _graded(grades)
    if metric == Metric.HIGH_MODERATE:
        mapped = [g for g in graded if g.certainty_status == AxisStatus.MAPPED]
        return sum(g.certainty in _HIGH_MODERATE for g in mapped), len(mapped)

    mapped = [g for g in graded if g.strength_status == AxisStatus.MAPPED]
    if metric == Metric.STRONG:
        return sum(g.strength == Strength.STRONG for g in mapped), len(mapped)
    return sum(g.direction == Direction.AGAINST for g in mapped), len(mapped)


# ── helpers ─────────────────────────────────────────────────────

def _proportion(num: int, den: int, ci: Optional[ClusterCi] = None) -> Proportion:
    if ci is None:
        return Proportion(num, den, _pct(num, den))
    return Proportion(num, den, _pct(num, den), ci.low, ci.high, ci.undefined_draws)


def _ratio(num: int, den: int) -> Optional[float]:
    return num / den if den else None


def _pct(num: int, den: int) -> Optional[float]:
    return _PCT * num / den if den else None


def _current(rows: Sequence[HarmonizedRow]) -> list[HarmonizedRow]:
    return [r for r in rows if r.edition == Edition.CURRENT]


def _graded(grades: Sequence[HarmonizedGrade]) -> list[HarmonizedGrade]:
    return [g for g in grades if g.category == Category.GRADED]


def _grades_by_doi(rows: Sequence[HarmonizedRow]) -> dict[str, list[HarmonizedGrade]]:
    out: dict[str, list[HarmonizedGrade]] = {}
    for r in rows:
        out.setdefault(r.doi, []).append(r.grade)
    return dict(sorted(out.items()))


def _groups(rows: list[HarmonizedRow]) -> list[tuple[str, list[HarmonizedRow]]]:
    """[(society, its rows)] sorted by name, then (ALL, every row); same order as analytics."""
    societies = sorted({r.society for r in rows})
    return [(s, [r for r in rows if r.society == s]) for s in societies] + [(ALL, rows)]
