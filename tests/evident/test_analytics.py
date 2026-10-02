"""Tests for evident.analytics: headline columns and denominator rule."""

import csv

from evident.analytics import HarmonizedRow, headline, write_headline
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


def test_write_atomic_with_empty_cells(tmp_path):
    path = tmp_path / "out" / "headline.csv"
    write_headline(headline([_ROWS[-1]], 1, "v", "m@d"), str(path))

    rows = list(csv.DictReader(open(path)))
    assert [r["society"] for r in rows] == ["ESICM", "ALL"]
    assert rows[0]["pct_strong"] == ""
    assert not (tmp_path / "out" / "headline.csv.tmp").exists()
