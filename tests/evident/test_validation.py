"""Tests for evident.validation: DOI join, scoring on harmonized grades, bootstrap gate."""

import dataclasses
import json

import pytest

from evident.domain import Category, GateResult, GradingFamily, RawRecommendation
from evident.validation import (
    GtRow,
    GuidelineScore,
    LabelledGuideline,
    SnapshotScore,
    ValidationJoinError,
    gate,
    load_labelled,
    score,
)
from tests.legacy_golden import FakeEncoder

_GRADED = Category.GRADED
_A = "ACP recommends against routine imaging in acute low back pain"
_B = "ACP suggests exercise therapy for chronic knee osteoarthritis"
_C = "Clinicians should assess fluid responsiveness before each bolus"
_U = "Best practice: involve patients in shared decision making"


def _rec(i, text, strength, certainty, category=_GRADED):
    return RawRecommendation(ordinal=i, text=text, raw_strength=strength, raw_certainty=certainty,
                             raw_category=category)


def _labelled(doi, rows):
    return LabelledGuideline(doi=doi, family=GradingFamily.GRADE, gt=[GtRow(*r) for r in rows])


_GT = _labelled("10.1/a", [
    (_A, "strong recommendation", "low-certainty evidence", _GRADED),
    (_B, "conditional recommendation", "moderate-certainty evidence", _GRADED),
    (_C, "strong recommendation", "high-certainty evidence", _GRADED),
    (_U, "0", "0", Category.BEST_PRACTICE),
])


class TestScore:
    def test_join_failure_names_missing_doi(self):
        other = _labelled("10.1/b", [])
        with pytest.raises(ValidationJoinError) as err:
            score({"10.1/a": []}, [_GT, other], FakeEncoder())
        assert err.value.missing_dois == ["10.1/b"]

    def test_doi_keys_normalized(self):
        result = score({"https://doi.org/10.1/A": []}, [_GT], FakeEncoder())
        assert result.per_guideline[0].fn == 3

    def test_counts_and_grade_axes(self):
        recs = [
            _rec(0, _A, "Strong", "Low"),            # match, against from text on both sides: all ok
            _rec(1, _B, "Strong", "Moderate"),       # match, strength wrong, certainty ok
            _rec(2, "Give vitamin Z to every adult with headache", "Strong", "High"),  # FP
            _rec(3, _U, "BEST_PRACTICE", "NA", Category.BEST_PRACTICE),  # ungraded TP
            _rec(4, "Good practice: wash hands", "BEST_PRACTICE", "NA", Category.BEST_PRACTICE),  # ungraded FP
        ]

        s = score({"10.1/a": recs}, [_GT], FakeEncoder()).per_guideline[0]

        assert (s.tp, s.fp, s.fn, s.n_matched) == (2, 1, 1, 2)
        assert (s.n_strength_ok, s.n_certainty_ok, s.n_combined_ok) == (1, 2, 1)
        assert (s.ungraded_tp, s.ungraded_fp, s.ungraded_fn) == (1, 1, 0)

    def test_label_for_overrides_against_text(self):
        # "Strong For" keeps FOR, GT text says against → strength axis wrong
        recs = [_rec(0, _A, "Strong For", "Low")]
        s = score({"10.1/a": recs}, [_GT], FakeEncoder()).per_guideline[0]
        assert (s.n_strength_ok, s.n_certainty_ok) == (0, 1)


def _scores(tp=5, fp=1, fn=1, combined=4, n=12, **overrides):
    """n guidelines with identical counts, then per-index overrides {i: dict(...)}."""
    rows = []
    for i in range(n):
        counts = {**dict(tp=tp, fp=fp, fn=fn, combined=combined), **overrides.get(f"g{i}", {})}
        rows.append(GuidelineScore(
            doi=f"10.1/{i:02d}", tp=counts["tp"], fp=counts["fp"], fn=counts["fn"], n_matched=counts["tp"],
            n_strength_ok=counts["combined"], n_certainty_ok=counts["combined"], n_combined_ok=counts["combined"],
            ungraded_tp=0, ungraded_fp=0, ungraded_fn=0))
    return SnapshotScore(tuple(rows))


class TestGate:
    def test_no_baseline(self):
        report = gate(_scores(), None)
        assert report.result == GateResult.NO_BASELINE
        assert report.candidate.f1 == pytest.approx(5 / 6)

    def test_identical_pass(self):
        report = gate(_scores(), _scores())
        assert report.result == GateResult.PASS
        assert report.f1_diff == 0 and report.f1_diff_ci == (0.0, 0.0)

    def test_clearly_worse_fails(self):
        assert gate(_scores(tp=2, fn=4, combined=0), _scores()).result == GateResult.FAIL

    def test_worse_on_f1_only_fails(self):
        # Fewer matches but every matched pair still fully correct
        worse = _scores(tp=3, fn=3, combined=3)
        report = gate(worse, _scores(combined=5))
        assert report.combined_diff == 0
        assert report.result == GateResult.FAIL

    def test_small_noise_passes(self):
        noisy = _scores(g3=dict(tp=4, fp=1, fn=2, combined=4))
        assert gate(noisy, _scores()).result == GateResult.PASS

    def test_same_seed_same_report(self):
        a, b = _scores(g0=dict(tp=1, fp=1, fn=5, combined=1)), _scores()
        assert gate(a, b, seed=3).to_json() == gate(a, b, seed=3).to_json()
        assert json.loads(gate(a, b).to_json())["result"] in {"pass", "fail"}

    def test_different_guidelines_refused(self):
        other = SnapshotScore(tuple(dataclasses.replace(g, doi="x" + g.doi) for g in _scores().per_guideline))
        with pytest.raises(ValueError):
            gate(_scores(), other)


def test_load_labelled_real_data():
    labelled = load_labelled()
    assert len(labelled) == 12
    assert sum(len(g.gt) for g in labelled) == 48
    assert all(g.doi == g.doi.lower() for g in labelled)
