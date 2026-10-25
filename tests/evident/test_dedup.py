"""Tests for evident.dedup: screen, adjudication file, clusters and the keep rule (spec §5 amendment)."""

import numpy as np
import pytest

from evident.analytics import HarmonizedRow
from evident.dedup import (
    HumanLabel,
    SPLIT_REASON,
    Adjudication,
    CandidatePair,
    DedupVerdictError,
    Verdict,
    check_complete,
    consensus,
    deduplicate,
    disagreements,
    load_human_check,
    score_human_check,
    lexical_ratio,
    load_adjudications,
    remap_ordinals,
    removed_rows,
    screen,
    write_adjudications,
)
from evident.domain import AxisStatus, Category, Certainty, Direction, HarmonizedGrade, Strength

_M, _UM = AxisStatus.MAPPED, AxisStatus.UNMAPPED
_STRONG_LOW = HarmonizedGrade(Category.GRADED, Strength.STRONG, Direction.FOR, Certainty.LOW, _M, _M)
_WEAK_LOW = HarmonizedGrade(Category.GRADED, Strength.WEAK, Direction.FOR, Certainty.LOW, _M, _M)
_NO_CERTAINTY = HarmonizedGrade(Category.GRADED, Strength.STRONG, Direction.FOR, None, _M, _UM)
_STRONG_HIGH = HarmonizedGrade(Category.GRADED, Strength.STRONG, Direction.FOR, Certainty.HIGH, _M, _M)


def _row(doi, ordinal, grade=_STRONG_LOW, society="SCCM", text="text"):
    return HarmonizedRow(doi, society, 2020, grade, text=text, page=1, ordinal=ordinal)


def _dup(doi, a, b):
    return Adjudication(doi, a, b, Verdict.DUPLICATE, "same PICO")


class TestLexicalRatio:
    def test_pdf_noise_is_identical(self):
        a = "ACP recommends adding an SGLT-2 inhibi- tor to met- formin (strong recommendation; high-certainty evidence)."
        b = "ACP recommends adding an SGLT-2 inhibitor to metformin"
        assert lexical_ratio(a, b) == 1.0

    def test_ligature_is_folded(self):
        assert lexical_ratio("lifestyle modiﬁcation", "lifestyle modification") == 1.0

    def test_other_population_is_lower(self):
        a = "We suggest balanced crystalloids rather than isotonic saline in adult critically ill patients"
        b = a + " with kidney injury"
        assert lexical_ratio(a, b) < 1.0


class TestScreen:
    _TEXTS = ["We recommend NMBA infusion in early ARDS",
              "We recommend neuromuscular-blocking agent infusion in early ARDS",
              "We suggest prone positioning for severe ARDS"]

    def test_either_criterion_flags_a_pair(self):
        sim = np.array([[1.0, 0.90, 0.30], [0.90, 1.0, 0.30], [0.30, 0.30, 1.0]])
        pairs = screen("d", [0, 1, 2], self._TEXTS, sim)
        assert [(p.ordinal_a, p.ordinal_b) for p in pairs] == [(0, 1)]
        assert pairs[0].biolord == pytest.approx(0.90)

    def test_lexical_alone_flags_a_pair(self):
        texts = ["We suggest prone positioning", "We suggest prone- positioning."]
        pairs = screen("d", [3, 7], texts, np.array([[1.0, 0.10], [0.10, 1.0]]))
        assert [(p.ordinal_a, p.ordinal_b) for p in pairs] == [(3, 7)]

    def test_below_both_is_not_flagged(self):
        texts = ["We recommend early enteral nutrition", "We suggest against routine stress ulcer prophylaxis"]
        assert screen("d", [0, 1], texts, np.full((2, 2), 0.5)) == []


class TestAdjudicationFile:
    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "verdicts.csv")
        pair = CandidatePair("d", 0, 1, 0.9, 0.95, "a", "b")
        write_adjudications(path, [(pair, Verdict.DUPLICATE, "restated", "m", "digest")])
        assert load_adjudications(path) == [Adjudication("d", 0, 1, Verdict.DUPLICATE, "restated", "m")]

    def test_complete_when_every_screened_pair_has_a_verdict(self):
        pairs = [CandidatePair("d", 0, 1, 0.9, 0.9, "a", "b")]
        check_complete(pairs, [_dup("d", 0, 1)])

    def test_missing_verdict_fails_closed(self):
        pairs = [CandidatePair("d", 0, 1, 0.9, 0.9, "a", "b")]
        with pytest.raises(DedupVerdictError, match="d rows 0, 1: no verdict"):
            check_complete(pairs, [])

    def test_verdict_for_unscreened_pair_fails_closed(self):
        with pytest.raises(DedupVerdictError, match="d rows 0, 2: not screened"):
            check_complete([], [_dup("d", 0, 2)])


class TestDeduplicate:
    def test_distinct_verdict_keeps_both(self):
        rows = [_row("d", 0), _row("d", 1)]
        result = deduplicate(rows, [Adjudication("d", 0, 1, Verdict.DISTINCT, "other population")])
        assert result.kept == rows
        assert result.removed == {}

    def test_lowest_ordinal_kept(self):
        rows = [_row("d", 0), _row("d", 1)]
        result = deduplicate(rows, [_dup("d", 0, 1)])
        assert [r.ordinal for r in result.kept] == [0]
        assert result.removed == {("d", 1): 0}

    def test_fully_mapped_copy_wins_over_lower_ordinal(self):
        rows = [_row("d", 0, _NO_CERTAINTY), _row("d", 1, _STRONG_LOW)]
        result = deduplicate(rows, [_dup("d", 0, 1)])
        assert [r.ordinal for r in result.kept] == [1]
        assert result.n_grade_conflicts == 1

    def test_chain_is_one_cluster(self):
        rows = [_row("d", 0), _row("d", 1), _row("d", 2)]
        result = deduplicate(rows, [_dup("d", 0, 1), _dup("d", 1, 2)])
        assert [r.ordinal for r in result.kept] == [0]
        assert result.removed == {("d", 1): 0, ("d", 2): 0}

    def test_same_ordinals_in_another_guideline_untouched(self):
        rows = [_row("d", 0), _row("d", 1), _row("e", 0), _row("e", 1)]
        result = deduplicate(rows, [_dup("d", 0, 1)])
        assert [(r.doi, r.ordinal) for r in result.kept] == [("d", 0), ("e", 0), ("e", 1)]

    def test_grade_conflict_counted_once_per_cluster(self):
        rows = [_row("d", 0, _NO_CERTAINTY), _row("d", 1, _STRONG_LOW), _row("d", 2, _STRONG_LOW)]
        result = deduplicate(rows, [_dup("d", 0, 1), _dup("d", 0, 2)])
        assert result.n_grade_conflicts == 1

    def test_different_mapped_grade_keeps_both(self):
        rows = [_row("d", 0, _STRONG_LOW), _row("d", 1, _WEAK_LOW)]
        result = deduplicate(rows, [_dup("d", 0, 1)])
        assert result.kept == rows
        assert result.n_blocked_by_grade == 1

    def test_missing_axis_does_not_join_conflicting_grades(self):
        # 0 low and 2 high both match 1 (no certainty): only the first merge stands
        rows = [_row("d", 0, _STRONG_LOW), _row("d", 1, _NO_CERTAINTY), _row("d", 2, _STRONG_HIGH)]
        result = deduplicate(rows, [_dup("d", 0, 1), _dup("d", 1, 2)])
        assert [r.ordinal for r in result.kept] == [0, 2]
        assert result.n_blocked_by_grade == 1

    def test_removed_by_society(self):
        rows = [_row("d", 0), _row("d", 1), _row("e", 0, society="ACP"), _row("e", 1, society="ACP")]
        result = deduplicate(rows, [_dup("d", 0, 1), _dup("e", 0, 1)])
        assert result.removed_by_society == {"ACP": 1, "SCCM": 1}


class TestExports:
    def test_removed_rows_name_the_kept_copy(self):
        rows = [_row("d", 0, _STRONG_LOW, text="kept"), _row("d", 1, _NO_CERTAINTY, text="copy")]
        (r,) = removed_rows(rows, deduplicate(rows, [_dup("d", 0, 1)]))
        assert (r.kept_ordinal, r.removed_ordinal, r.same_grade, r.kept_text, r.removed_text) == (0, 1, False, "kept",
                                                                                                    "copy")

    def test_remap_points_removed_ordinals_to_kept(self):
        assert remap_ordinals("d", (3, 1), {("d", 3): 0, ("e", 1): 0}) == (0, 1)


class TestConsensus:
    def _judge(self, model, *verdicts):
        return [Adjudication("d", i, i + 1, v, f"{model} reason", model) for i, v in enumerate(verdicts)]

    def test_agreement_is_the_verdict(self):
        a = self._judge("qwen", Verdict.DUPLICATE, Verdict.DISTINCT)
        b = self._judge("gemma", Verdict.DUPLICATE, Verdict.DISTINCT)
        assert disagreements(a, b) == []
        assert consensus(a, b) == [Adjudication("d", 0, 1, Verdict.DUPLICATE, "qwen reason", "qwen+gemma"),
                                       Adjudication("d", 1, 2, Verdict.DISTINCT, "qwen reason", "qwen+gemma")]

    def test_disagreement_keeps_both(self):
        a = self._judge("qwen", Verdict.DUPLICATE, Verdict.DUPLICATE)
        b = self._judge("gemma", Verdict.DISTINCT, Verdict.DUPLICATE)
        assert disagreements(a, b) == [("d", 0, 1)]
        assert consensus(a, b)[0] == Adjudication("d", 0, 1, Verdict.DISTINCT, SPLIT_REASON, "qwen+gemma")

    def test_judges_must_cover_the_same_pairs(self):
        a = self._judge("qwen", Verdict.DUPLICATE, Verdict.DISTINCT)
        b = self._judge("gemma", Verdict.DUPLICATE)
        with pytest.raises(DedupVerdictError, match="d rows 1, 2: judged by one model only"):
            consensus(a, b)


class TestHumanCheck:
    def test_load_joins_labels_and_key(self, tmp_path):
        sample, key = tmp_path / "sample.csv", tmp_path / "key.csv"
        sample.write_text("id,doi,text_a,text_b,label,note\n1,d,a,b,Same ,\n2,d,a,c,different,\n")
        key.write_text("id,stratum,doi,ordinal_a,ordinal_b,verdict\n1,s1,d,0,1,duplicate\n2,s2,d,0,2,distinct\n")
        assert load_human_check(str(sample), str(key)) == [HumanLabel("d", 0, 1, "s1", True),
                                                           HumanLabel("d", 0, 2, "s2", False)]

    def test_unlabelled_pair_fails_closed(self, tmp_path):
        sample, key = tmp_path / "sample.csv", tmp_path / "key.csv"
        sample.write_text("id,doi,text_a,text_b,label,note\n1,d,a,b,,\n")
        key.write_text("id,stratum,doi,ordinal_a,ordinal_b,verdict\n1,s1,d,0,1,duplicate\n")
        with pytest.raises(DedupVerdictError, match="human check id 1: label"):
            load_human_check(str(sample), str(key))

    def test_scored_against_the_final_clusters(self):
        rows = [_row("d", 0), _row("d", 1), _row("d", 2), _row("d", 3), _row("d", 4, _WEAK_LOW)]
        removed = {("d", 1): 0}  # 0–1 merged
        labels = [HumanLabel("d", 0, 1, "merged", True),     # agree
                  HumanLabel("d", 2, 3, "split", True),      # missed duplicate
                  HumanLabel("d", 0, 2, "split", False),     # agree
                  HumanLabel("d", 3, 4, "merged", True)]     # same text, other grade → different: agree
        check = score_human_check(labels, rows, removed)
        assert (check.n, check.n_agree, check.n_false_merge, check.n_missed) == (4, 3, 0, 1)
        assert check.by_stratum == {"merged": {"n": 2, "n_agree": 2}, "split": {"n": 2, "n_agree": 1}}
        assert 0 < check.agree_ci_low < 0.75 < check.agree_ci_high < 1

    def test_false_merge_counted(self):
        rows = [_row("d", 0), _row("d", 1)]
        check = score_human_check([HumanLabel("d", 0, 1, "merged", False)], rows, {("d", 1): 0})
        assert (check.n_agree, check.n_false_merge, check.agree_ci_low) == (0, 1, 0.0)
