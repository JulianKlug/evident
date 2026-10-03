"""Table-driven tests for evident.harmonization (GRADE family)."""

import csv
import os

import pytest

from evident.domain import AxisStatus, Category, Certainty, Direction, GradingFamily, Strength
from evident.harmonization import UnsupportedGradingFamilyError, harmonize
from extraction.datasets import GtMode, load_acp_datasets, load_icu_datasets

_EXPECTED_CSV = os.path.join(os.path.dirname(__file__), "..", "fixtures", "harmonization_gt_expected.csv")
_N_GT_ROWS = 48  # ACP 29 + ICU 19 (2 ESICM part 2 trauma recs added 2026-10-03)
S, W = Strength.STRONG, Strength.WEAK
F, A = Direction.FOR, Direction.AGAINST
M, U, NG = AxisStatus.MAPPED, AxisStatus.UNMAPPED, AxisStatus.UNGRADED
G = Category.GRADED


def _gt_rows():
    """(key, raw_class, raw_LOE, text, category) for every ACP + ICU GT row."""
    datasets = load_acp_datasets(GtMode.WITH_UNGRADED) + load_icu_datasets(GtMode.WITH_UNGRADED)
    return [
        (f"{ds.doi}#{i}", r["raw_class"], r["raw_LOE"], r["recommendation"], Category(r["category"]))
        for ds in datasets for i, r in ds.ground_truth_df.iterrows()
    ]


def _expected():
    with open(_EXPECTED_CSV, newline="") as f:
        return {row.pop("key"): row for row in csv.DictReader(f)}


def _as_row(h):
    value = lambda e: e.value if e else ""
    return {"category": h.category.value, "strength": value(h.strength), "direction": value(h.direction),
            "certainty": value(h.certainty), "strength_status": h.strength_status.value,
            "certainty_status": h.certainty_status.value}


def _assert_invariant(h):
    assert (h.strength_status == M) == (h.strength is not None)
    assert (h.certainty_status == M) == (h.certainty is not None)


GT_ROWS = _gt_rows()


def test_every_gt_row_has_an_expectation():
    assert len(GT_ROWS) == _N_GT_ROWS
    assert {r[0] for r in GT_ROWS} == set(_expected())


@pytest.mark.parametrize("key, raw_class, raw_loe, text, category", GT_ROWS, ids=[r[0] for r in GT_ROWS])
def test_gt_row(key, raw_class, raw_loe, text, category):
    h = harmonize(raw_class, raw_loe, text, category, GradingFamily.GRADE)

    assert _as_row(h) == _expected()[key]
    _assert_invariant(h)


# (raw_strength, raw_certainty, text, raw_category) → (category, strength, direction, certainty, s_status, c_status)
SYNTHETIC = [
    # empty everywhere → graded, nothing mapped
    (("", "", "", G), (G, None, None, None, U, U)),
    (("nan", "0.0", "Give X", G), (G, None, None, None, U, U)),
    # unknown tokens
    (("moderate recommendation", "medium", "Give X", G), (G, None, None, None, U, U)),
    # insufficient certainty is ungraded, not unmapped
    (("Strong", "insufficient evidence", "Give X", G), (G, S, F, None, M, NG)),
    # label direction overrides text
    (("Strong For", "High", "We suggest against X", G), (G, S, F, Certainty.HIGH, M, M)),
    (("Weak against", "Low", "We suggest X", G), (G, W, A, Certainty.LOW, M, M)),
    # "rather than" / "over" compare options, both FOR
    (("Conditional", "Low", "We suggest A rather than B", G), (G, W, F, Certainty.LOW, M, M)),
    (("Conditional", "Low", "We suggest A over B", G), (G, W, F, Certainty.LOW, M, M)),
    # text against patterns
    (("Strong", "High", "Clinicians should not give X", G), (G, S, A, Certainty.HIGH, M, M)),
    (("Strong", "High", "Do not administer X", G), (G, S, A, Certainty.HIGH, M, M)),
    # truncation repair; ambiguous suffix (several certainty forms) stays unmapped
    (("trong recommendation", "igh-certainty evidence", "Give X", G), (G, S, F, Certainty.HIGH, M, M)),
    (("Strong", "ertainty", "Give X", G), (G, S, F, None, M, U)),
    (("Strong", "ow", "Give X", G), (G, S, F, None, M, U)),
    # composite certainty → lower
    (("Strong", "moderate/low-certainty evidence", "Give X", G), (G, S, F, Certainty.LOW, M, M)),
    (("Strong", "high/very low", "Give X", G), (G, S, F, Certainty.VERY_LOW, M, M)),
    (("Strong", "Moderate quality of evidence", "Give X", G), (G, S, F, Certainty.MODERATE, M, M)),
    (("Strong", "low confidence in estimates of effect", "Give X", G), (G, S, F, Certainty.LOW, M, M)),
    # categories from label, text, or parser
    (("No recommendation", "Moderate", "X", G), (Category.NO_RECOMMENDATION, None, None, None, NG, NG)),
    (("", "", "We cannot recommend for or against X", G), (Category.NO_RECOMMENDATION, None, None, None, NG, NG)),
    (("Best practice statement", "", "Reassess", G), (Category.BEST_PRACTICE, None, None, None, NG, NG)),
    (("", "", "Good practice statement: reassess", G), (Category.BEST_PRACTICE, None, None, None, NG, NG)),
    (("BEST_PRACTICE", "NA", "Reassess", Category.BEST_PRACTICE), (Category.BEST_PRACTICE, None, None, None, NG, NG)),
]


@pytest.mark.parametrize("args, expected", SYNTHETIC)
def test_synthetic(args, expected):
    h = harmonize(*args, family=GradingFamily.GRADE)

    assert (h.category, h.strength, h.direction, h.certainty, h.strength_status, h.certainty_status) == expected
    _assert_invariant(h)


def test_non_grade_family_raises():
    with pytest.raises(UnsupportedGradingFamilyError):
        harmonize("I", "A", "Give X", G, "esc_ers")
