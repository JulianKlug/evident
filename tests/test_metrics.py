import pytest
import pandas as pd

from evaluation.matching import MatchResult, RecommendationMatch
from evaluation.metrics import compute_metrics
from evaluation.grading import GradingScheme


def _make_match(ext_grade="I", ext_level="A", gt_grade="I", gt_level="A", score=0.9):
    return RecommendationMatch(
        extracted_text="text",
        extracted_grade=ext_grade,
        extracted_level=ext_level,
        gt_text="text",
        gt_grade=gt_grade,
        gt_level=gt_level,
        similarity_score=score,
    )


_empty_fp = pd.DataFrame(columns=["recommendation", "class", "LOE"])
_empty_fn = pd.DataFrame(columns=["recommendation", "class", "LOE"])


def _fp_rows(n):
    return pd.DataFrame({
        "recommendation": [f"fp_{i}" for i in range(n)],
        "class": ["I"] * n,
        "LOE": ["A"] * n,
    })


def _fn_rows(n):
    return pd.DataFrame({
        "recommendation": [f"fn_{i}" for i in range(n)],
        "class": ["I"] * n,
        "LOE": ["A"] * n,
    })


class TestPrecisionRecallF1:
    def test_all_matched(self):
        matches = [_make_match() for _ in range(5)]
        result = compute_metrics(MatchResult(matches, _empty_fp, _empty_fn))
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)
        assert result.f1 == pytest.approx(1.0)
        assert result.n_tp == 5
        assert result.n_fp == 0
        assert result.n_fn == 0

    def test_with_fp_and_fn(self):
        matches = [_make_match() for _ in range(3)]
        result = compute_metrics(MatchResult(matches, _fp_rows(2), _fn_rows(1)))
        # precision = 3 / (3+2) = 0.6
        assert result.precision == pytest.approx(0.6)
        # recall = 3 / (3+1) = 0.75
        assert result.recall == pytest.approx(0.75)
        # f1 = 2 * 0.6 * 0.75 / (0.6 + 0.75)
        assert result.f1 == pytest.approx(2 * 0.6 * 0.75 / 1.35)

    def test_no_matches(self):
        result = compute_metrics(MatchResult([], _fp_rows(2), _fn_rows(3)))
        assert result.precision == pytest.approx(0.0)
        assert result.recall == pytest.approx(0.0)
        assert result.f1 == pytest.approx(0.0)

    def test_no_fp_no_fn_no_matches(self):
        """Edge case: empty everything."""
        result = compute_metrics(MatchResult([], _empty_fp, _empty_fn))
        assert result.precision == pytest.approx(0.0)
        assert result.recall == pytest.approx(0.0)
        assert result.f1 == pytest.approx(0.0)

    def test_single_match(self):
        result = compute_metrics(MatchResult([_make_match()], _empty_fp, _empty_fn))
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)
        assert result.f1 == pytest.approx(1.0)


class TestGradeLevelAccuracy:
    def test_all_correct(self):
        matches = [_make_match(ext_grade="I", gt_grade="I", ext_level="A", gt_level="A") for _ in range(4)]
        result = compute_metrics(MatchResult(matches, _empty_fp, _empty_fn))
        assert result.grade_accuracy == pytest.approx(1.0)
        assert result.level_accuracy == pytest.approx(1.0)
        assert result.combined_accuracy == pytest.approx(1.0)

    def test_mixed_accuracy(self):
        matches = [
            _make_match(ext_grade="I", gt_grade="I", ext_level="A", gt_level="A"),     # both correct
            _make_match(ext_grade="IIa", gt_grade="I", ext_level="A", gt_level="A"),   # grade wrong
            _make_match(ext_grade="I", gt_grade="I", ext_level="B", gt_level="A"),     # level wrong
            _make_match(ext_grade="IIa", gt_grade="I", ext_level="B", gt_level="A"),   # both wrong
        ]
        result = compute_metrics(MatchResult(matches, _empty_fp, _empty_fn))
        assert result.grade_accuracy == pytest.approx(2 / 4)
        assert result.level_accuracy == pytest.approx(2 / 4)
        assert result.combined_accuracy == pytest.approx(1 / 4)

    def test_case_insensitive(self):
        matches = [_make_match(ext_grade="i", gt_grade="I", ext_level="a", gt_level="A")]
        result = compute_metrics(MatchResult(matches, _empty_fp, _empty_fn))
        assert result.grade_accuracy == pytest.approx(1.0)
        assert result.level_accuracy == pytest.approx(1.0)

    def test_no_matches_accuracy(self):
        result = compute_metrics(MatchResult([], _fp_rows(1), _fn_rows(1)))
        assert result.grade_accuracy == pytest.approx(0.0)
        assert result.level_accuracy == pytest.approx(0.0)
        assert result.combined_accuracy == pytest.approx(0.0)


class TestWithGradingScheme:
    def test_normalization(self, simple_scheme):
        """Aliases should be normalized before comparison."""
        matches = [_make_match(ext_grade="1", gt_grade="I", ext_level="A", gt_level="A")]
        result = compute_metrics(MatchResult(matches, _empty_fp, _empty_fn), grading_scheme=simple_scheme)
        # "1" normalizes to "I" via alias, so grade should match
        assert result.grade_accuracy == pytest.approx(1.0)

    def test_normalization_mismatch(self, simple_scheme):
        matches = [_make_match(ext_grade="2a", gt_grade="I", ext_level="A", gt_level="A")]
        result = compute_metrics(MatchResult(matches, _empty_fp, _empty_fn), grading_scheme=simple_scheme)
        # "2a" → "IIa", "I" → "I" → mismatch
        assert result.grade_accuracy == pytest.approx(0.0)


class TestConfusionMatrix:
    def test_structure(self):
        matches = [
            _make_match(ext_grade="I", gt_grade="I"),
            _make_match(ext_grade="IIa", gt_grade="I"),
            _make_match(ext_grade="I", gt_grade="IIa"),
        ]
        result = compute_metrics(MatchResult(matches, _empty_fp, _empty_fn))
        assert not result.grade_confusion.empty

    def test_per_class(self):
        matches = [
            _make_match(ext_grade="I", gt_grade="I"),
            _make_match(ext_grade="I", gt_grade="I"),
            _make_match(ext_grade="IIa", gt_grade="IIa"),
            _make_match(ext_grade="I", gt_grade="IIa"),  # wrong
        ]
        result = compute_metrics(MatchResult(matches, _empty_fp, _empty_fn))
        # GT "i" has 2 correct out of 2 → 1.0
        assert result.per_class_grade["i"] == pytest.approx(1.0)
        # GT "iia" has 1 correct out of 2 → 0.5
        assert result.per_class_grade["iia"] == pytest.approx(0.5)
