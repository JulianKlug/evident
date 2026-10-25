"""Tests for evident.validation: DOI join, scoring on harmonized grades, bootstrap gate."""

import dataclasses
import json
import os

import pytest

from evident.domain import AxisStatus, Category, Certainty, Direction, GateResult, GradingFamily, RawRecommendation, Strength
from evident.validation import (
    GtRow,
    GuidelineScore,
    HeldoutLabelError,
    LabelledGuideline,
    SnapshotScore,
    ValidationJoinError,
    gate,
    heldout_report,
    load_heldout,
    load_labelled,
    load_spotcheck,
    score,
)
from tests.evident.pipeline_fakes import write_heldout_xlsx
from tests.legacy_golden import FakeEncoder

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


# ── held-out labels ─────────────────────────────────────────────

_H1, _H2 = "10.1097/ccm.1", "10.1097/pcc.2"
_HELD_ROWS = [
    (_A, "strong recommendation, against", "low certainty evidence", 3),
    (_B, "conditional recommendation, for", "moderate certainty evidence", 4.0),
    (_U, "Best practice statement", "NA", 4),
]


class TestLoadHeldout:
    def _dir(self, tmp_path, rows=None):
        directory = str(tmp_path / "heldout")
        write_heldout_xlsx(directory, _H1, _HELD_ROWS if rows is None else rows)
        write_heldout_xlsx(directory, _H2, _HELD_ROWS[:1])
        return directory

    def _problems(self, directory, expected=(_H1, _H2), labelled=frozenset()):
        with pytest.raises(HeldoutLabelError) as e:
            load_heldout(directory, expected, set(labelled))
        return e.value.problems

    def test_complete_set_loads_with_categories(self, tmp_path):
        labelled = load_heldout(self._dir(tmp_path), [_H1.upper(), _H2], set())
        assert [g.doi for g in labelled] == [_H1, _H2]
        assert [r.category for r in labelled[0].gt] == [Category.GRADED, Category.GRADED, Category.GRADED]
        assert labelled[0].gt[0].raw_class == "strong recommendation, against"

    def test_missing_and_unexpected_files(self, tmp_path):
        problems = self._problems(self._dir(tmp_path), expected=(_H1, "10.1/missing"))
        assert "10.1/missing: no label file" in problems
        assert any("10.1097/pcc.2 is not a prespecified" in p for p in problems)

    def test_development_label_refused(self, tmp_path):
        problems = self._problems(self._dir(tmp_path), labelled={_H1})
        assert any("also a development label" in p for p in problems)

    def test_empty_template_refused(self, tmp_path):
        problems = self._problems(self._dir(tmp_path, rows=[]))
        assert any("no labelled statement" in p for p in problems)

    @pytest.mark.parametrize("row, message", [
        ((_A, "strong recommendation", "low", "p3"), "is not a page number"),
        ((_A, "", "low", 3), "empty class"),
        ((_A, "strong recommendation", None, 3), "empty LOE"),
    ])
    def test_incomplete_row_refused(self, tmp_path, row, message):
        problems = self._problems(self._dir(tmp_path, rows=[row]))
        assert any(f"line 2: {message}" in p or message in p for p in problems), problems

    def test_evidence_statements_not_scored(self, tmp_path):
        rows = [_HELD_ROWS[0], ("ICU patients have more ADEs", " Evidence statement ", "B", None)]
        labelled = load_heldout(self._dir(tmp_path, rows=rows), [_H1, _H2], set())
        assert [r.text for r in labelled[0].gt] == [_A]

    def test_blank_page_allowed(self, tmp_path):
        rows = [(_A, "strong recommendation", "low", None)]
        assert [g.doi for g in load_heldout(self._dir(tmp_path, rows=rows), [_H1, _H2], set())] == [_H1, _H2]

    def test_duplicate_statement_refused(self, tmp_path):
        rows = [_HELD_ROWS[0], (" " + _A.upper() + " ", "strong recommendation", "low", 5)]
        problems = self._problems(self._dir(tmp_path, rows=rows))
        assert any("line 3: duplicate of line 2" in p for p in problems)

    def test_corrupt_file_reported(self, tmp_path):
        directory = self._dir(tmp_path)
        with open(os.path.join(directory, "bad.xlsx"), "wb") as f:
            f.write(b"not a zip")
        assert any("bad.xlsx: unreadable" in p for p in self._problems(directory))

    def test_excel_lock_file_ignored(self, tmp_path):
        directory = self._dir(tmp_path)
        with open(os.path.join(directory, "~$10.1097_ccm.1.xlsx"), "wb") as f:
            f.write(b"lock")
        assert [g.doi for g in load_heldout(directory, [_H1, _H2], set())] == [_H1, _H2]

    def test_labels_sheet_must_come_first(self, tmp_path):
        directory = self._dir(tmp_path)
        write_heldout_xlsx(directory, _H2, _HELD_ROWS, sheet="Sheet1")
        problems = self._problems(directory)
        assert any("first sheet must be 'labels'" in p for p in problems)

    def test_not_a_directory(self, tmp_path):
        assert self._problems(str(tmp_path / "absent")) == [f"{tmp_path / 'absent'}: not a directory"]


class TestHeldoutReport:
    def test_micro_pooled_with_wilson_intervals(self):
        s = SnapshotScore((
            GuidelineScore("a", tp=9, fp=1, fn=0, n_matched=9, n_strength_ok=9, n_certainty_ok=8, n_combined_ok=8,
                           ungraded_tp=0, ungraded_fp=0, ungraded_fn=2),
            GuidelineScore("b", tp=0, fp=0, fn=1, n_matched=0, n_strength_ok=0, n_certainty_ok=0, n_combined_ok=0,
                           ungraded_tp=0, ungraded_fp=0, ungraded_fn=0),
        ))
        r = heldout_report(s)
        assert (r.n_guidelines, r.tp, r.fp, r.fn) == (2, 9, 1, 1)
        assert r.precision == pytest.approx(0.9) and r.recall == pytest.approx(0.9)
        assert r.precision_ci == pytest.approx((0.59585, 0.98212), abs=1e-5)
        assert r.combined_accuracy == pytest.approx(8 / 9)
        assert (r.ungraded_precision, r.ungraded_recall) == (None, 0.0)
        assert "ignores clustering" in r.interval_note

    def test_empty_denominators(self):
        r = heldout_report(SnapshotScore(()))
        assert (r.precision, r.f1, r.precision_ci) == (None, None, (None, None))


# ── M3 spot check ───────────────────────────────────────────────

def test_load_spotcheck(tmp_path):
    data = {"doi": "10.7326/M16-1", "matches": [{"ref_id": "R2", "extracted_id": 4}, {"ref_id": "R1", "extracted_id": 0}],
            "reference": [
                {"category": "graded", "strength": "strong", "direction": "against", "certainty": "high"},
                {"category": "graded", "strength": "weak", "direction": "neither", "certainty": "mixed: low/very_low"},
                {"category": "best_practice", "strength": "", "direction": "for", "certainty": ""},
            ]}
    (tmp_path / "10.7326_m16-1.spotcheck.json").write_text(json.dumps(data))
    (tmp_path / "notes.txt").write_text("ignored")

    [g] = load_spotcheck(str(tmp_path))
    assert (g.doi, g.matched_ordinals) == ("10.7326/m16-1", (0, 4))
    strong, mixed, bps = g.reference
    assert (strong.strength, strong.direction, strong.certainty) == (Strength.STRONG, Direction.AGAINST, Certainty.HIGH)
    assert (mixed.direction, mixed.strength_status, mixed.certainty_status) == (None, AxisStatus.MAPPED,
                                                                                AxisStatus.UNMAPPED)
    assert (bps.category, bps.strength_status) == (Category.BEST_PRACTICE, AxisStatus.UNGRADED)


def test_committed_spotcheck_reproduces_reference_counts():
    """corpus/spotcheck_m3: 217 graded and 160 ungraded reference statements (docs/progress.md)."""
    spot = load_spotcheck(os.path.join(_REPO, "corpus", "spotcheck_m3"))
    grades = [r for g in spot for r in g.reference]
    assert len(spot) == 10
    assert sum(r.category == Category.GRADED for r in grades) == 217
    assert sum(r.category != Category.GRADED for r in grades) == 160
