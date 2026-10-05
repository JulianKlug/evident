"""Tests for evident.corpus: manifest validation, current editions, store sync."""

import os

import pytest

from evident.corpus import (
    CandidateDecision,
    ExclusionReason,
    ManifestError,
    check_candidates,
    current_editions,
    load_candidates,
    load_manifest,
    sync_to_store,
)
from evident.store import DuplicatePdfError, Store

_REPO_MANIFEST = os.path.join(os.path.dirname(__file__), "..", "..", "corpus", "manifest.csv")
_REPO_CANDIDATES = os.path.join(os.path.dirname(__file__), "..", "..", "corpus", "candidates.csv")
_N_REPO_GUIDELINES = 74
_HEADER = "doi,society,year,title,topic_id,supersedes_doi,grading_family,pdf_filename"
_A = "10.1007/a,ESICM,2016,Fluids v1,fluids,,grade,"
_B = "10.1007/b,ESICM,2020,Fluids v2,fluids,10.1007/a,grade,"
_C = "10.1007/c,ESICM,2024,Fluids v3,fluids,10.1007/b,grade,"
_ACP = "https://doi.org/10.7326/M22-2056,ACP,2023,Depression,depression,,GRADE,"


def _manifest(tmp_path, *rows, header=_HEADER):
    path = tmp_path / "manifest.csv"
    path.write_text("\n".join([header, *rows]) + "\n")
    return str(path)


def _problems(path):
    with pytest.raises(ManifestError) as err:
        load_manifest(path)
    return err.value.problems


class TestLoadManifest:
    def test_repo_manifest_is_valid(self):
        entries = load_manifest(_REPO_MANIFEST)
        assert len(entries) == _N_REPO_GUIDELINES
        assert {e.society for e in entries} == {"ACP", "ESICM", "SCCM"}
        assert current_editions(entries) == {e.doi for e in entries} - {e.supersedes_doi for e in entries}

    def test_repo_candidates_match_manifest(self):
        check_candidates(load_candidates(_REPO_CANDIDATES), load_manifest(_REPO_MANIFEST))

    def test_round_trip(self, tmp_path):
        entries = load_manifest(_manifest(tmp_path, _A, _B, _ACP))

        acp = entries[2]
        assert (acp.doi, acp.year, acp.grading_family.value) == ("10.7326/m22-2056", 2023, "grade")
        assert acp.pdf_filename == "10_7326_M22-2056.pdf"  # original case, as on disk
        assert entries[1].supersedes_doi == "10.1007/a"
        assert entries[0].supersedes_doi is None

    def test_missing_column(self, tmp_path):
        assert "line 1" in _problems(_manifest(tmp_path, header=_HEADER.replace(",topic_id", "")))[0]

    @pytest.mark.parametrize("row, fragment", [
        ("nodoi,ESICM,2016,T,t,,grade,", "invalid doi"),
        ("10.1007/a,ESC,2016,T,t,,grade,", "unknown society"),
        ("10.1007/a,ESICM,1989,T,t,,grade,", "year"),
        ("10.1007/a,ESICM,2999,T,t,,grade,", "year"),
        ("10.1007/a,ESICM,abc,T,t,,grade,", "year"),
        ("10.1007/a,ESICM,2016,,t,,grade,", "empty title"),
        ("10.1007/a,ESICM,2016,T,Not A Slug,,grade,", "slug"),
        ("10.1007/a,ESICM,2016,T,t,,esc_ers,", "grading_family"),
        ("10.1007/a,ESICM,2016,T,t,10.1007/zz,grade,", "unknown doi"),
    ])
    def test_rule_fails_with_line_number(self, tmp_path, row, fragment):
        problems = _problems(_manifest(tmp_path, row))
        assert len(problems) == 1
        assert problems[0].startswith("line 2:") and fragment in problems[0]

    def test_joint_with(self, tmp_path):
        path = _manifest(tmp_path, _A + ",ACP; SCCM", header=_HEADER + ",joint_with")
        assert load_manifest(path)[0].joint_with == ("ACP", "SCCM")
        assert load_manifest(_manifest(tmp_path, _A))[0].joint_with == ()

    @pytest.mark.parametrize("joint", ["ESC", "ESICM"])
    def test_joint_with_must_be_another_known_society(self, tmp_path, joint):
        problems = _problems(_manifest(tmp_path, f"{_A},{joint}", header=_HEADER + ",joint_with"))
        assert "joint_with" in problems[0]

    def test_all_problems_reported_together(self, tmp_path):
        problems = _problems(_manifest(tmp_path, "bad,ESC,1,,X,,nope,", _A, _A))
        assert sum(p.startswith("line 2:") for p in problems) == 6
        assert any(p.startswith("line 4:") and "duplicate" in p for p in problems)

    def test_cross_topic_supersede(self, tmp_path):
        other = "10.1007/b,ESICM,2020,Other,sepsis,10.1007/a,grade,"
        assert "another topic" in " ".join(_problems(_manifest(tmp_path, _A, other)))

    def test_supersede_must_be_earlier(self, tmp_path):
        same_year = "10.1007/b,ESICM,2016,Fluids v2,fluids,10.1007/a,grade,"
        assert "not earlier" in " ".join(_problems(_manifest(tmp_path, _A, same_year)))

    def test_cycle(self, tmp_path):
        a = "10.1007/a,ESICM,2016,Fluids v1,fluids,10.1007/b,grade,"
        assert "cycle" in " ".join(_problems(_manifest(tmp_path, a, _B)))


class TestCurrentEditions:
    def test_chain(self, tmp_path):
        assert current_editions(load_manifest(_manifest(tmp_path, _A, _B, _C, _ACP))) == {"10.1007/c", "10.7326/m22-2056"}

    def test_latest_year_without_supersedes(self, tmp_path):
        b = "10.1007/b,ESICM,2020,Fluids v2,fluids,,grade,"
        assert current_editions(load_manifest(_manifest(tmp_path, _A, b))) == {"10.1007/b"}

    def test_same_year_tie(self, tmp_path):
        b = "10.1007/b,ESICM,2016,Fluids alt,fluids,,grade,"
        with pytest.raises(ManifestError, match="ambiguous"):
            current_editions(load_manifest(_manifest(tmp_path, _A, b)))


class TestSyncToStore:
    def test_added_then_updated_and_missing_pdf(self, tmp_path):
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        (pdf_dir / "10_1007_a.pdf").write_bytes(b"%PDF a")
        entries = load_manifest(_manifest(tmp_path, _A, _B))

        with Store.open(str(tmp_path / "db.sqlite")) as store:
            first = sync_to_store(entries, store, str(pdf_dir))
            second = sync_to_store(entries, store, str(pdf_dir))

            assert first.added == ["10.1007/a", "10.1007/b"] and first.updated == []
            assert second.updated == ["10.1007/a", "10.1007/b"] and second.added == []
            assert first.missing_pdf == ["10.1007/b"]
            assert store.get_guideline("10.1007/a").pdf_sha256
            assert store.get_guideline("10.1007/b").pdf_sha256 is None

    def test_duplicate_pdf_raises(self, tmp_path):
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        for name in ("10_1007_a.pdf", "10_1007_b.pdf"):
            (pdf_dir / name).write_bytes(b"%PDF same bytes")
        b = "10.1007/b,ESICM,2020,Fluids v2,fluids,,grade,"

        with Store.open(str(tmp_path / "db.sqlite")) as store:
            with pytest.raises(DuplicatePdfError):
                sync_to_store(load_manifest(_manifest(tmp_path, _A, b)), store, str(pdf_dir))


_CANDIDATE_HEADER = "doi,society,year,title,journal,pmid,source,decision,reason,duplicate_of"
_INCLUDE_A = "10.1007/a,ESICM,2016,Fluids v1,Intensive Care Med,1,pubmed,include,,"
_INCLUDE_B = "10.1007/b,ESICM,2020,Fluids v2,Intensive Care Med,2,pubmed,include,,"


def _candidates(tmp_path, *rows):
    path = tmp_path / "candidates.csv"
    path.write_text("\n".join([_CANDIDATE_HEADER, *rows]) + "\n")
    return str(path)


def _candidate_problems(path):
    with pytest.raises(ManifestError) as err:
        load_candidates(path)
    return err.value.problems


class TestCandidates:
    def test_round_trip(self, tmp_path):
        dup = "10.1097/x,SCCM,2021,Copy,Crit Care Med,3,pubmed,exclude,duplicate_publication,10.1007/A"
        a, b, c = load_candidates(_candidates(tmp_path, _INCLUDE_A, _INCLUDE_B, dup))
        assert (a.decision, a.reason) == (CandidateDecision.INCLUDE, None)
        assert (c.reason, c.duplicate_of) == (ExclusionReason.DUPLICATE_PUBLICATION, "10.1007/a")

    @pytest.mark.parametrize("row, fragment", [
        ("10.1007/x,ESICM,2016,T,J,1,pubmed,maybe,,", "decision"),
        ("10.1007/x,ESICM,2016,T,J,1,pubmed,exclude,,", "needs a reason"),
        ("10.1007/x,ESICM,2016,T,J,1,pubmed,exclude,dislike,", "needs a reason"),
        ("10.1007/x,ESICM,2016,T,J,1,pubmed,include,not_graded,", "must not have a reason"),
        ("10.1007/x,ESICM,2016,T,J,1,pubmed,exclude,duplicate_publication,10.1007/zz", "duplicate_of"),
        ("bad,ESICM,2016,T,J,1,pubmed,include,,", "invalid doi"),
    ])
    def test_rule_fails_with_line_number(self, tmp_path, row, fragment):
        problems = _candidate_problems(_candidates(tmp_path, row))
        assert len(problems) == 1
        assert problems[0].startswith("line 2:") and fragment in problems[0]

    def test_check_against_manifest(self, tmp_path):
        manifest = load_manifest(_manifest(tmp_path, _A, _B))
        check_candidates(load_candidates(_candidates(tmp_path, _INCLUDE_A, _INCLUDE_B)), manifest)

        excluded_b = _INCLUDE_B.replace("include,", "exclude,not_graded")
        with pytest.raises(ManifestError) as err:
            check_candidates(load_candidates(_candidates(tmp_path, _INCLUDE_A, excluded_b)), manifest)
        assert err.value.problems == ["manifest row 10.1007/b is not an included candidate"]

        with pytest.raises(ManifestError) as err:
            check_candidates(load_candidates(_candidates(tmp_path, _INCLUDE_A, _INCLUDE_B)), manifest[:1])
        assert err.value.problems == ["included candidate 10.1007/b has no manifest row"]
