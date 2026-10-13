"""Tests for evident.paper: denominators, editions, guideline mean, cluster bootstrap, corpus flow, spot check.

Fixtures are small enough to check by hand (spec §9 criteria 2–5).
"""

import pytest

from evident.analytics import Edition, HarmonizedRow
from evident.corpus import Candidate, CandidateDecision, ExclusionReason, ManifestEntry
from evident.domain import AxisStatus, Category, Certainty, Direction, GradingFamily, HarmonizedGrade, Strength
from evident.paper import (
    ALL,
    Metric,
    cluster_ci,
    corpus_flow,
    excluded_guidelines,
    figure_data,
    guideline_mean,
    raw_mapping,
    spotcheck_bias,
    table1,
    table2,
    trend_points,
)
from evident.validation import SpotcheckGuideline

_M, _UM, _UG = AxisStatus.MAPPED, AxisStatus.UNMAPPED, AxisStatus.UNGRADED
_S, _W = Strength.STRONG, Strength.WEAK
_FOR, _AGAINST = Direction.FOR, Direction.AGAINST
_H, _MOD, _L, _VL = Certainty.HIGH, Certainty.MODERATE, Certainty.LOW, Certainty.VERY_LOW
_N_BOOT, _SEED = 200, 0


def _g(strength, direction, certainty):
    return HarmonizedGrade(Category.GRADED, strength, direction, certainty,
                           _M if strength else _UM, _M if certainty else _UM)


def _ungraded(category=Category.BEST_PRACTICE):
    return HarmonizedGrade(category, None, None, None, _UG, _UG)


def _row(doi, society, grade, year=2020, edition=Edition.CURRENT, ordinal=None, raw=("raw s", "raw c")):
    return HarmonizedRow(doi, society, year, grade, edition, "text", 1, raw[0], raw[1], ordinal)


def _t1(rows):
    return {r.society: r for r in table1(rows, _N_BOOT, _SEED)}


# ── denominators (criterion 2) ──────────────────────────────────

class TestDenominators:
    _ROWS = [
        _row("a", "ACP", _g(_S, _AGAINST, _L)),       # strong against: strong AND against
        _row("a", "ACP", _g(_W, _FOR, _H)),
        _row("a", "ACP", _g(None, None, _MOD)),       # strength unmapped: out of strong/against only
        _row("a", "ACP", _g(_S, _FOR, None)),         # certainty unmapped: out of high/moderate only
        _row("a", "ACP", _ungraded()),                # best practice: in no denominator
        _row("a", "ACP", _ungraded(Category.NO_RECOMMENDATION)),  # "no suggestion for or against"
    ]

    def test_axis_specific_denominators(self):
        acp = _t1(self._ROWS)["ACP"]
        assert (acp.n_recs, acp.n_graded) == (6, 4)
        assert (acp.strong.num, acp.strong.den) == (2, 3)
        assert (acp.against.num, acp.against.den) == (1, 3)
        assert (acp.high_moderate.num, acp.high_moderate.den) == (2, 3)
        assert acp.strong.pct == pytest.approx(200 / 3)  # unrounded
        assert (acp.n_strength_unmapped, acp.n_certainty_unmapped) == (1, 1)

    def test_no_recommendation_never_against(self):
        acp = _t1(self._ROWS)["ACP"]
        assert acp.against.num == 1  # only the strong-against row

    def test_empty_denominator_is_none_not_zero(self):
        rows = [_row("a", "ACP", _ungraded()), _row("b", "SCCM", _g(_S, _FOR, _H))]
        acp = _t1(rows)["ACP"]
        assert (acp.strong.den, acp.strong.pct) == (0, None)
        assert (acp.strong.ci_low, acp.strong.ci_high) == (None, None)
        assert acp.strong.undefined_draws == _N_BOOT

    def test_all_is_pooled_per_recommendation(self):
        rows = [_row("a", "ACP", _g(_S, _FOR, _H))] + [_row("b", "SCCM", _g(_W, _FOR, _L)) for _ in range(9)]
        t1 = _t1(rows)
        assert t1[ALL].strong.pct == pytest.approx(10.0)  # 1/10, not mean(100, 0) = 50
        assert [r.society for r in table1(rows, _N_BOOT, _SEED)] == ["ACP", "SCCM", ALL]

    def test_table2_certainty_and_crosstab(self):
        rows = [_row("a", "ACP", _g(_S, _FOR, _H)), _row("a", "ACP", _g(_S, _FOR, _L)),
                _row("a", "ACP", _g(_W, _FOR, _L)), _row("a", "ACP", _g(None, None, _L))]
        t2 = {(r.society, r.table, r.certainty): r for r in table2(rows)}
        assert (t2["ACP", "certainty", _L].n, t2["ACP", "certainty", _L].den) == (3, 4)
        assert (t2["ACP", "strong", _L].n, t2["ACP", "strong", _L].den) == (1, 2)
        assert t2["ACP", "strong", _L].pct == pytest.approx(50.0)
        assert (t2["ACP", "weak", _VL].n, t2["ACP", "weak", _VL].den) == (0, 1)


# ── editions (criterion 3) ──────────────────────────────────────

class TestEditions:
    _ROWS = [
        _row("old", "ESICM", _g(_W, _FOR, _VL), year=2016, edition=Edition.SUPERSEDED),
        _row("old", "ESICM", _g(_W, _FOR, _VL), year=2016, edition=Edition.SUPERSEDED),
        _row("new", "ESICM", _g(_S, _FOR, _H), year=2024),
    ]

    def test_superseded_absent_from_cross_section(self):
        esicm = _t1(self._ROWS)["ESICM"]
        assert (esicm.n_guidelines, esicm.strong.num, esicm.strong.den) == (1, 1, 1)
        assert all(r.society != "ESICM" or r.den in (0, 1) for r in table2(self._ROWS))
        assert {r.n_guidelines for r in guideline_mean(self._ROWS)} == {1}

    def test_superseded_present_in_trends(self):
        points = {(p.society, p.year, p.metric): p for p in trend_points(self._ROWS)}
        old = points["ESICM", 2016, Metric.STRONG]
        assert (old.n_guidelines, old.n_graded, old.num, old.den) == (1, 2, 0, 2)
        assert points[ALL, 2024, Metric.HIGH_MODERATE].pct == pytest.approx(100.0)

    def test_joint_guideline_counted_once(self):
        """A co-published guideline is one manifest row (the copy is a duplicate_publication exclusion)."""
        rows = [_row("joint", "SCCM", _g(_S, _FOR, _H))]
        t1 = _t1(rows)
        assert (t1["SCCM"].n_guidelines, t1[ALL].n_guidelines, t1[ALL].strong.den) == (1, 1, 1)


# ── guideline mean (criterion 4) ────────────────────────────────

class TestGuidelineMean:
    def test_unequal_sizes_differ_from_pooled(self):
        # A: 1/1 strong; B: 1/9 strong → mean (100 + 11.1)/2 = 55.6; pooled 2/10 = 20.0
        rows = ([_row("a", "ACP", _g(_S, _FOR, _H))]
                + [_row("b", "ACP", _g(_S if i == 0 else _W, _FOR, _L)) for i in range(9)])
        strong = next(r for r in guideline_mean(rows) if (r.society, r.metric) == ("ACP", Metric.STRONG))
        assert strong.mean_pct == pytest.approx((100 + 100 / 9) / 2)
        assert strong.pooled_pct == pytest.approx(20.0)
        assert strong.diff_pp == pytest.approx((100 + 100 / 9) / 2 - 20.0)
        assert (strong.n_guidelines, strong.n_guidelines_omitted) == (2, 0)

    def test_zero_denominator_guideline_omitted_and_counted(self):
        rows = [_row("a", "ACP", _g(_S, _FOR, None)), _row("b", "ACP", _g(_W, _FOR, _H))]
        hm = next(r for r in guideline_mean(rows) if (r.society, r.metric) == ("ACP", Metric.HIGH_MODERATE))
        assert (hm.n_guidelines, hm.n_guidelines_omitted, hm.mean_pct) == (1, 1, 100.0)

    def test_no_contributing_guideline(self):
        rows = [_row("a", "ACP", _ungraded())]
        strong = next(r for r in guideline_mean(rows) if (r.society, r.metric) == ("ACP", Metric.STRONG))
        assert (strong.mean_pct, strong.pooled_pct, strong.diff_pp) == (None, None, None)


# ── cluster bootstrap (criterion 5) ─────────────────────────────

class TestClusterCi:
    _CLUSTERS = [("ACP", 1, 2), ("ACP", 3, 4), ("ACP", 0, 5), ("SCCM", 10, 20), ("SCCM", 1, 10)]

    def test_deterministic_for_a_seed(self):
        assert cluster_ci(self._CLUSTERS, 500, 0) == cluster_ci(self._CLUSTERS, 500, 0)

    def test_single_cluster_has_no_spread(self):
        ci = cluster_ci([("ACP", 1, 4)], 100, 0)
        assert (ci["ACP"].low, ci["ACP"].high) == (25.0, 25.0)

    def test_resamples_guidelines_not_recommendations(self):
        # Two guidelines at 0% and 100%: draws give 0, 50 or 100 → wide; independent recs would be narrow
        ci = cluster_ci([("ACP", 0, 50), ("ACP", 50, 50)], 2000, 0)["ACP"]
        assert (ci.low, ci.high) == (0.0, 100.0)

    def test_all_is_stratified_by_society(self):
        # Each society is one fixed guideline; stratified draws can't change the mix → no spread
        ci = cluster_ci([("ACP", 1, 2), ("SCCM", 3, 4)], 200, 0)[ALL]
        assert (ci.low, ci.high) == (pytest.approx(400 / 6), pytest.approx(400 / 6))

    def test_undefined_draws_dropped_and_counted(self):
        ci = cluster_ci([("ACP", 0, 0), ("ACP", 1, 1)], 400, 0)["ACP"]
        assert 0 < ci.undefined_draws < 400  # draws of the empty guideline twice
        assert (ci.low, ci.high) == (100.0, 100.0)

    def test_table1_cis_bracket_the_estimate(self):
        rows = ([_row(f"a{i}", "ACP", _g(_S if i % 2 else _W, _FOR, _H)) for i in range(6)]
                + [_row(f"s{i}", "SCCM", _g(_S if i % 3 == 0 else _W, _FOR, _L)) for i in range(9)])
        for r in table1(rows, 500, 0):
            assert r.strong.ci_low <= r.strong.pct <= r.strong.ci_high


# ── figure data ─────────────────────────────────────────────────

def test_figure_data_covers_every_figure():
    rows = [_row("a", "ACP", _g(_S, _FOR, _H)), _row("b", "SCCM", _g(_W, _AGAINST, _L), year=2018)]
    t1, t2, trends = table1(rows, 50, 0), table2(rows), trend_points(rows)
    points = figure_data(t1, t2, trends)
    by_figure = {p.figure for p in points}
    assert by_figure == {"fig1_certainty_by_society", "fig2_strong_by_society", "fig3_strength_by_certainty",
                         "fig4_trends"}
    fig2 = [p for p in points if p.figure == "fig2_strong_by_society"]
    assert [(p.society, p.num, p.den) for p in fig2] == [("ACP", 1, 1), ("SCCM", 0, 1), (ALL, 1, 2)]
    assert len([p for p in points if p.figure == "fig4_trends"]) == len(trends)


# ── corpus ──────────────────────────────────────────────────────

def _candidate(doi, decision, reason=None, society="ACP", source="pubmed"):
    return Candidate(doi, society, 2020, f"T {doi}", decision, reason, None, source)


def _entry(doi, society):
    return ManifestEntry(doi, society, 2020, "T", "topic-" + doi, GradingFamily.GRADE, doi + ".pdf")


def test_corpus_flow_lists_every_reason_and_society():
    candidates = [_candidate("a", CandidateDecision.INCLUDE), _candidate("b", CandidateDecision.INCLUDE),
                  _candidate("c", CandidateDecision.EXCLUDE, ExclusionReason.NOT_GUIDELINE, source="pubmed_broad"),
                  _candidate("d", CandidateDecision.EXCLUDE, ExclusionReason.NOT_GUIDELINE)]
    entries = [_entry("a", "ACP"), _entry("b", "SCCM")]
    flow = {(r.stage, r.item): r.n for r in corpus_flow(candidates, entries, {"a"})}

    assert flow["candidates", "all"] == 4
    assert (flow["source", "pubmed"], flow["source", "pubmed_broad"]) == (3, 1)
    assert flow["excluded", "not_guideline"] == 2
    assert flow["excluded", "pdf_unavailable"] == 0  # zero-count reason still listed
    assert (flow["included", "all"], flow["included", "SCCM"]) == (2, 1)
    assert (flow["current", "all"], flow["current", "ACP"], flow["superseded", "all"]) == (1, 1, 1)
    assert ("current", "SCCM") not in flow


def test_excluded_guidelines_sorted_by_reason_then_doi():
    candidates = [_candidate("z", CandidateDecision.EXCLUDE, ExclusionReason.DERIVATIVE),
                  _candidate("b", CandidateDecision.EXCLUDE, ExclusionReason.NOT_GUIDELINE),
                  _candidate("a", CandidateDecision.EXCLUDE, ExclusionReason.NOT_GUIDELINE),
                  _candidate("i", CandidateDecision.INCLUDE)]
    assert [(r.doi, r.reason) for r in excluded_guidelines(candidates)] == [
        ("z", "derivative"), ("a", "not_guideline"), ("b", "not_guideline")]


def test_raw_mapping_counts_editions():
    rows = [_row("a", "ACP", _g(_S, _FOR, _H), raw=("Strong", "High")),
            _row("a", "ACP", _g(_S, _FOR, _H), raw=("Strong", "High")),
            _row("o", "ACP", _g(_S, _FOR, _H), raw=("Strong", "High"), edition=Edition.SUPERSEDED),
            _row("a", "ACP", _g(None, None, _H), raw=("Strng?", "High"))]
    first, second = raw_mapping(rows)
    assert (first.raw_strength, first.n_all_editions, first.n_current) == ("Strong", 3, 2)
    assert (second.raw_strength, second.strength_status, second.n_all_editions) == ("Strng?", _UM, 1)


# ── spot check ──────────────────────────────────────────────────

class TestSpotcheckBias:
    # Extracted: 0 strong (matched), 1 weak (matched), 2 strong restatement (FP), 3 best practice (FP)
    _ROWS = [_row("a", "SCCM", _g(_S, _FOR, _H), ordinal=0), _row("a", "SCCM", _g(_W, _FOR, _L), ordinal=1),
             _row("a", "SCCM", _g(_S, _FOR, _H), ordinal=2), _row("a", "SCCM", _ungraded(), ordinal=3)]
    _SPOT = [SpotcheckGuideline("a", (0, 1), (_g(_S, _FOR, _H), _g(_W, _FOR, _L), _g(_W, _AGAINST, _L),
                                              _ungraded(), _ungraded()))]

    def _bias(self):
        return {r.metric: r for r in spotcheck_bias(self._ROWS, self._SPOT, 100, 0)}

    def test_published_correct_truth(self):
        strong = self._bias()[Metric.STRONG]
        assert (strong.published.num, strong.published.den) == (2, 3)
        assert (strong.correct.num, strong.correct.den) == (1, 2)
        assert (strong.truth.num, strong.truth.den) == (1, 3)
        assert strong.diff_pp == pytest.approx(100 / 3)
        assert strong.diff_ci_low == strong.diff_ci_high == pytest.approx(100 / 3)  # one cluster

    def test_ungraded_share_has_no_correct_column(self):
        share = self._bias()[Metric.UNGRADED_SHARE]
        assert (share.published.num, share.published.den, share.truth.num, share.truth.den) == (1, 4, 2, 5)
        assert share.correct is None

    def test_missing_doi_refused(self):
        with pytest.raises(ValueError, match="b"):
            spotcheck_bias(self._ROWS, [SpotcheckGuideline("b", (), ())], 10, 0)
