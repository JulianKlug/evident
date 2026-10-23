"""Tests for evident.themes: certainty per assigned theme, specialties, order, unassigned recs."""

from evident.analytics import Edition, HarmonizedRow
from evident.domain import AxisStatus, Category, Certainty, Direction, HarmonizedGrade, Strength
from evident.themes import OTHER, UNASSIGNED, ThemeRow, rec_key, theme_certainty

_M, _UG = AxisStatus.MAPPED, AxisStatus.UNGRADED


def _rec(society, text, certainty, doi="10.1/a", edition=Edition.CURRENT, status=_M):
    grade = HarmonizedGrade(Category.GRADED, Strength.WEAK, Direction.FOR, certainty, _M, status)
    return HarmonizedRow(doi, society, 2022, grade, edition, text)


def _assigned(pairs):
    """[(rec, theme)] → (rows, assignments)."""
    return [r for r, _ in pairs], {rec_key(r.doi, r.text): t for r, t in pairs}


def _by_theme(out, specialty="Critical care"):
    themes = {}
    for r in out:
        if r.specialty == specialty:
            themes.setdefault((r.theme, r.label), {})[r.certainty] = r.pct
    return themes


def test_rec_key_depends_on_doi_and_text_not_doi_case():
    assert rec_key("10.1/A", "Use X") == rec_key("10.1/a", "Use X")
    assert rec_key("10.1/a", "Use X") != rec_key("10.1/a", "Use Y")
    assert rec_key("10.1/a", "Use X") != rec_key("10.1/b", "Use X")


def test_societies_of_a_specialty_share_its_themes():
    rows, assignments = _assigned([(_rec("SCCM", "a", Certainty.HIGH, "10.1/s"), "Fluids"),
                                   (_rec("ESICM", "b", Certainty.LOW, "10.1/e"), "Fluids")])

    out = theme_certainty(rows, assignments)

    assert {(r.specialty, r.societies, r.label, r.n_recs, r.n_guidelines) for r in out} == {
        ("Critical care", ("ESICM", "SCCM"), "Fluids", 2, 2)}
    assert _by_theme(out)[(1, "Fluids")] == {Certainty.HIGH: 50.0, Certainty.MODERATE: 0.0,
                                             Certainty.LOW: 50.0, Certainty.VERY_LOW: 0.0}


def test_specialties_are_separate():
    rows, assignments = _assigned([(_rec("SCCM", "a", Certainty.HIGH), "Fluids"),
                                   (_rec("ACP", "b", Certainty.LOW, "10.1/p"), "Gout")])

    out = theme_certainty(rows, assignments)

    assert {(r.specialty, r.label) for r in out} == {("Critical care", "Fluids"), ("Internal medicine", "Gout")}


def test_ordered_by_mean_certainty_other_then_unassigned_last():
    rows, assignments = _assigned([(_rec("SCCM", "a", Certainty.VERY_LOW), "Nutrition"),
                                   (_rec("SCCM", "b", Certainty.HIGH), OTHER),
                                   (_rec("SCCM", "c", Certainty.LOW), "Fluids"),
                                   (_rec("SCCM", "d", Certainty.HIGH), "Glucose control")])
    rows.append(_rec("SCCM", "new", Certainty.HIGH))  # extracted after the file was frozen

    order = sorted(_by_theme(theme_certainty(rows, assignments)))

    assert [label for _, label in order] == ["Glucose control", "Fluids", "Nutrition", OTHER, UNASSIGNED]


def test_denominator_is_certainty_mapped_current_recs():
    rows, assignments = _assigned([(_rec("ACP", "a", Certainty.HIGH), "Gout"),
                                   (_rec("ACP", "b", None, status=_UG), "Gout"),
                                   (_rec("ACP", "c", Certainty.LOW, edition=Edition.SUPERSEDED), "Gout")])

    out = theme_certainty(rows, assignments)

    assert out[0] == ThemeRow("Internal medicine", ("ACP",), 1, "Gout", 2, 1, Certainty.HIGH, 1, 100.0)


def test_theme_without_certainty_mapped_recs_has_empty_shares():
    rows, assignments = _assigned([(_rec("ACP", "a", None, status=_UG), "Gout")])

    assert {r.pct for r in theme_certainty(rows, assignments)} == {None}


def test_empty_rows():
    assert theme_certainty([], {}) == []
