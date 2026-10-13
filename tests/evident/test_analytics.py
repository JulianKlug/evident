"""Tests for evident.analytics: headline columns, denominator rule, distributions, trends, captions."""

from evident.analytics import (
    Edition,
    GuidelineFacts,
    HarmonizedRow,
    caption_facts,
    certainty_distribution,
    guideline_summary,
    headline,
    strength_by_certainty,
    trends,
)
from evident.domain import AxisStatus, Category, Certainty, Direction, HarmonizedGrade, Strength

_M, _UM, _UG = AxisStatus.MAPPED, AxisStatus.UNMAPPED, AxisStatus.UNGRADED


def _graded(strength, direction, certainty, s_status=_M, c_status=_M):
    return HarmonizedGrade(Category.GRADED, strength, direction, certainty, s_status, c_status)


_UNGRADED = HarmonizedGrade(Category.BEST_PRACTICE, None, None, None, _UG, _UG)

_ROWS = [
    HarmonizedRow("10.1/a", "ACP", 2023, _graded(Strength.STRONG, Direction.FOR, Certainty.HIGH)),
    HarmonizedRow("10.1/a", "ACP", 2023, _graded(Strength.WEAK, Direction.AGAINST, Certainty.LOW)),
    HarmonizedRow("10.1/a", "ACP", 2023, _graded(None, None, None, s_status=_UM, c_status=_UM)),
    HarmonizedRow("10.1/b", "ACP", 2024, _graded(Strength.STRONG, Direction.AGAINST, None, c_status=_UG)),
    HarmonizedRow("10.1/c", "ESICM", 2025, _UNGRADED),
]


def test_every_column():
    acp, esicm, total = headline(_ROWS, snapshot_id=3, extractor_version_id="v", model="m@d")

    assert (acp.society, acp.n_guidelines, acp.n_recs, acp.n_graded, acp.n_ungraded) == ("ACP", 2, 4, 4, 0)
    assert acp.ungraded_share == 0.0
    # Unmapped strength row excluded: 2 strong of 3 mapped
    assert (acp.pct_strong, acp.pct_against) == (66.7, 66.7)
    # Certainty: HIGH + LOW mapped; one unmapped, one ungraded (insufficient) excluded
    assert acp.pct_high_moderate == 50.0
    assert (acp.n_strength_unmapped, acp.n_certainty_unmapped, acp.n_certainty_ungraded) == (1, 1, 1)
    assert (acp.snapshot_id, acp.extractor_version_id, acp.model) == (3, "v", "m@d")

    assert (esicm.n_recs, esicm.n_ungraded, esicm.ungraded_share) == (1, 1, 100.0)
    assert total.society == "ALL" and total.ungraded_share == 20.0


def test_empty_denominators_are_empty_not_zero():
    (esicm, _) = headline([_ROWS[-1]], 1, "v", "m@d")
    assert (esicm.pct_strong, esicm.pct_against, esicm.pct_high_moderate) == (None, None, None)


def test_all_is_sum_of_societies():
    *societies, total = headline(_ROWS, 1, "v", "m@d")
    for column in ("n_guidelines", "n_recs", "n_graded", "n_ungraded", "n_strength_unmapped",
                   "n_certainty_unmapped", "n_certainty_ungraded"):
        assert getattr(total, column) == sum(getattr(s, column) for s in societies)


def test_headline_skips_superseded_editions():
    old = HarmonizedRow("10.1/old", "ACP", 2018, _graded(Strength.WEAK, Direction.FOR, Certainty.LOW),
                        edition=Edition.SUPERSEDED)
    acp, _, total = headline(_ROWS + [old], 1, "v", "m@d")
    assert (acp.n_guidelines, acp.n_recs, total.n_recs) == (2, 4, 5)


def test_certainty_distribution():
    rows = {(r.society, r.certainty): r for r in certainty_distribution(_ROWS)}
    # ACP: HIGH + LOW mapped; unmapped and ungraded certainty are not in the denominator
    assert (rows["ACP", Certainty.HIGH].n, rows["ACP", Certainty.HIGH].pct) == (1, 50.0)
    assert (rows["ACP", Certainty.MODERATE].n, rows["ACP", Certainty.MODERATE].pct) == (0, 0.0)
    # ESICM has no graded rec: empty, not 0
    assert rows["ESICM", Certainty.HIGH].pct is None
    assert rows["ALL", Certainty.LOW].n == 1


def test_strength_by_certainty_row_percent():
    rows = {(r.society, r.strength, r.certainty): r for r in strength_by_certainty(_ROWS)}
    strong_high = rows["ACP", Strength.STRONG, Certainty.HIGH]
    # The strong rec with ungraded certainty has no certainty: only 1 strong rec in the crosstab
    assert (strong_high.n, strong_high.pct_of_strength) == (1, 100.0)
    assert rows["ACP", Strength.WEAK, Certainty.LOW].pct_of_strength == 100.0
    assert sum(r.n for r in strength_by_certainty(_ROWS) if r.society == "ALL") == 2


def test_trends_keep_superseded_editions():
    old = HarmonizedRow("10.1/old", "ACP", 2018, _graded(Strength.STRONG, Direction.FOR, Certainty.HIGH),
                        edition=Edition.SUPERSEDED)
    rows = {(r.society, r.year): r for r in trends(_ROWS + [old])}
    assert (rows["ACP", 2018].n_recs, rows["ACP", 2018].pct_strong) == (1, 100.0)
    assert (rows["ACP", 2023].n_guidelines, rows["ACP", 2023].n_recs, rows["ACP", 2023].pct_strong) == (1, 3, 50.0)
    assert rows["ESICM", 2025].pct_strong is None
    assert sorted(y for s, y in rows if s == "ALL") == [2018, 2023, 2024, 2025]


_FACTS = [
    GuidelineFacts("10.1/a", "ACP", 2023, "A", "t-a", Edition.CURRENT),
    GuidelineFacts("10.1/b", "ACP", 2024, "B", "t-b", Edition.CURRENT),
    GuidelineFacts("10.1/c", "ESICM", 2025, "C", "t-c", Edition.CURRENT, joint_with=("SCCM",)),
    GuidelineFacts("10.1/d", "SCCM", 2021, "D", "t-d", Edition.CURRENT, excluded_reason="scanned PDF"),
]


def test_guideline_summary():
    rows = guideline_summary(_ROWS, _FACTS)
    assert [r.doi for r in rows] == ["10.1/a", "10.1/b", "10.1/c", "10.1/d"]
    a, _, c, d = rows
    assert (a.n_recs, a.n_graded, a.pct_strong) == (3, 3, 50.0)
    assert (c.joint_with, c.ungraded_share) == ("SCCM", 100.0)
    assert (d.n_recs, d.excluded_reason, d.ungraded_share) == (0, "scanned PDF", None)


def test_caption_facts():
    facts = caption_facts(_ROWS, _FACTS, labelled_dois={"10.1/a", "10.1/c"})
    assert facts.societies == ("ACP", "ESICM", "SCCM")
    assert (facts.n_guidelines, facts.n_recs, facts.n_graded) == (3, 5, 4)
    assert (facts.n_strength_unmapped, facts.n_certainty_unmapped, facts.n_certainty_ungraded) == (1, 1, 1)
    assert facts.excluded == (("10.1/d", "scanned PDF"),)
    assert facts.joint == (("10.1/c", "ESICM", ("SCCM",)),)
    assert facts.unvalidated_societies == ("SCCM",)
    assert (facts.year_min, facts.year_max) == (2023, 2025)


def test_caption_facts_all_validated():
    facts = caption_facts(_ROWS, _FACTS[:3], labelled_dois={"10.1/a", "10.1/c"})
    assert facts.unvalidated_societies == ()
