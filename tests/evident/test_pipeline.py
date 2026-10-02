"""Tests for evident.pipeline with a tmp DB, tmp manifest and fake deps (no GPU)."""

import csv
import dataclasses
import json
import os

import pytest

from evident.domain import GateResult, MemberOrigin, RunStatus, SnapshotState, ThinkingMode
from evident.extraction import ArtifactMismatchError
from evident.pipeline import (
    AddStatus,
    AmbiguousPublishError,
    CoverageError,
    GateFailedError,
    IngestMode,
    ManifestEntryMissingError,
    NoPublishedSnapshotError,
    NotPublishableError,
    NoSnapshotInProgressError,
    NothingToPublishError,
    SnapshotInProgressError,
    SnapshotStart,
    StaleValidationError,
    ValidationMissingError,
    add,
    publish,
    reject,
    snapshot,
    status,
    validate,
)
from evident.store import DuplicatePdfError, InvalidSnapshotTransitionError
from evident.validation import ValidationJoinError
from tests.evident.pipeline_fakes import CONFIG, WORSE_CONFIG, FakeRunner, World

_A, _B, _C, _D = "10.1000/a", "10.1000/b", "10.1000/c", "10.1000/d"
_NEW, _RESUME = SnapshotStart.NEW, SnapshotStart.RESUME


@pytest.fixture
def world(tmp_path):
    w = World(tmp_path)
    yield w
    w.store.close()


def _published(world):
    """v0 published over the 3 manifest rows."""
    snapshot(world.deps(), _NEW, CONFIG)
    validate(world.deps())
    publish(world.deps())
    return world.store.latest_snapshot(SnapshotState.PUBLISHED)


class TestSnapshot:
    def test_new_extracts_all_and_completes(self, world):
        report = snapshot(world.deps(), _NEW, CONFIG)

        assert report.state == SnapshotState.COMPLETE
        assert (report.n_succeeded, report.n_failed, report.pending) == (3, 0, [])
        assert world.runner.calls == [_A, _B, _C]
        run = world.store.active_run(report.snapshot_id, world.store.get_guideline(_A).id)
        assert (run.code_sha, run.thinking, json.loads(run.few_shot_json)) == (
            "sha1", ThinkingMode.OFF, [{"recommendation": "x"}])

    def test_failing_runner_keeps_building_and_continues(self, world):
        world.runner = FakeRunner(fail=(_B,))

        report = snapshot(world.deps(), _NEW, CONFIG)

        assert report.state == SnapshotState.BUILDING
        assert world.runner.calls == [_A, _B, _C]
        assert report.pending == [_B]
        assert report.failures[0][0] == _B and "ollama crashed" in report.failures[0][1]

    def test_new_refused_while_building(self, world):
        world.runner = FakeRunner(fail=(_B,))
        snapshot(world.deps(), _NEW, CONFIG)
        with pytest.raises(SnapshotInProgressError):
            snapshot(world.deps(), _NEW, CONFIG)

    def test_resume_without_building(self, world):
        with pytest.raises(NoSnapshotInProgressError):
            snapshot(world.deps(), _RESUME)

    def test_resume_after_crash_reruns_only_pending(self, world):
        def crash_on_second(n):
            if n == 2:
                raise KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            snapshot(world.deps(runner=FakeRunner(on_call=crash_on_second)), _NEW, CONFIG)
        assert status(world.deps()).n_interrupted == 1

        report = snapshot(world.deps(), _RESUME)

        assert world.runner.calls == [_B, _C]
        assert report.state == SnapshotState.COMPLETE
        assert status(world.deps()).n_interrupted == 1  # history kept

    def test_excludes_unblock_complete(self, world):
        snapshot(world.deps(runner=FakeRunner(fail=(_C,))), _NEW, CONFIG)

        report = snapshot(world.deps(runner=FakeRunner(fail=(_C,))), _RESUME, excludes=[(_C, "scanned image")])

        assert report.state == SnapshotState.COMPLETE
        assert status(world.deps()).excluded == [(_C, "scanned image")]

    def test_missing_pdf_fails_and_is_not_excluded(self, world):
        os.remove(os.path.join(world.pdf_dir, "10_1000_c.pdf"))

        report = snapshot(world.deps(), _NEW, CONFIG)

        assert report.state == SnapshotState.BUILDING
        assert report.failures[0][0] == _C and "PdfMissingError" in report.failures[0][1]
        assert _C not in world.runner.calls
        assert status(world.deps()).excluded == []

    def test_rescan_picks_up_row_added_during_build(self, world):
        def add_row_on_first_call(n):
            if n == 1:
                world.add_row(_D)
        world.runner = FakeRunner(on_call=add_row_on_first_call)

        report = snapshot(world.deps(), _NEW, CONFIG)

        assert world.runner.calls == [_A, _B, _C, _D]
        assert report.state == SnapshotState.COMPLETE
        assert world.store.members(report.snapshot_id)[-1].origin == MemberOrigin.RESCAN

    def test_artifact_drift_on_resume_refused(self, world):
        snapshot(world.deps(runner=FakeRunner(fail=(_A,))), _NEW, CONFIG)
        world.probe.artifacts = dataclasses.replace(world.probe.artifacts, model_digest="re-pulled")
        n_runs = len(world.store.runs(1))

        with pytest.raises(ArtifactMismatchError):
            snapshot(world.deps(), _RESUME)
        assert len(world.store.runs(1)) == n_runs


class TestValidate:
    def test_first_has_no_baseline(self, world):
        snapshot(world.deps(), _NEW, CONFIG)
        report = validate(world.deps())
        assert report.gate.result == GateResult.NO_BASELINE
        assert report.gate.candidate.f1 == 1.0
        assert world.store.latest_validation(report.snapshot_id).gate == GateResult.NO_BASELINE

    def test_join_failure(self, world):
        snapshot(world.deps(runner=FakeRunner(fail=(_A,))), _NEW, CONFIG)
        # A labelled guideline excluded: the snapshot completes but cannot be scored
        world.store.exclude_member(1, world.store.get_guideline(_A).id, "test")
        world.store.transition_snapshot(1, SnapshotState.COMPLETE)
        with pytest.raises(ValidationJoinError):
            validate(world.deps())

    def test_second_snapshot_compared_to_published(self, world):
        published = _published(world)
        snapshot(world.deps(), _NEW, WORSE_CONFIG)

        report = validate(world.deps())

        assert report.baseline_snapshot_id == published.id
        assert report.gate.result == GateResult.FAIL


class TestPublish:
    def test_nothing_to_publish(self, world):
        with pytest.raises(NothingToPublishError):
            publish(world.deps())

    def test_missing_validation(self, world):
        snapshot(world.deps(), _NEW, CONFIG)
        with pytest.raises(ValidationMissingError):
            publish(world.deps())

    def test_publish_writes_headline(self, world):
        published = _published(world)

        rows = list(csv.DictReader(open(world.headline)))
        assert [r["society"] for r in rows] == ["ACP", "ESICM", "ALL"]
        assert rows[-1]["n_guidelines"] == "3" and rows[-1]["snapshot_id"] == str(published.id)

    def test_fail_refused_then_reason_stored(self, world):
        _published(world)
        second = snapshot(world.deps(), _NEW, WORSE_CONFIG).snapshot_id
        validate(world.deps())

        with pytest.raises(GateFailedError):
            publish(world.deps())
        publish(world.deps(), accept_regression="smaller model, accepted for speed")

        assert world.store.get_snapshot(second).accept_reason == "smaller model, accepted for speed"

    def test_coverage_error(self, world):
        snapshot(world.deps(), _NEW, CONFIG)
        validate(world.deps())
        world.add_row(_D)
        with pytest.raises(CoverageError) as err:
            publish(world.deps())
        assert err.value.dois == [_D]

    def test_republish_and_superseded_skipped(self, world):
        _published(world)
        world.add_row(_D)
        add(world.deps(), os.path.join(world.pdf_dir, "10_1000_d.pdf"))

        report = publish(world.deps())

        assert report.republished and report.n_headline_guidelines == 4
        assert list(csv.DictReader(open(world.headline)))[-1]["n_guidelines"] == "4"

    def test_superseded_edition_not_counted(self, tmp_path):
        w = World(tmp_path, dois=(_A,))
        w.add_row(_D, supersedes=_A, topic="topic-a")  # same topic, later year
        snapshot(w.deps(), _NEW, CONFIG)
        w.store.save_validation(1, None, GateResult.NO_BASELINE, "{}")

        report = publish(w.deps())

        w.store.close()
        assert (report.n_headline_guidelines, report.n_skipped_superseded) == (1, 1)


class TestAdd:
    def test_needs_published_snapshot(self, world):
        with pytest.raises(NoPublishedSnapshotError):
            add(world.deps(), os.path.join(world.pdf_dir, "10_1000_a.pdf"))

    def test_manifest_row_missing(self, world):
        _published(world)
        pdf = world.write_pdf(_D)
        with pytest.raises(ManifestEntryMissingError):
            add(world.deps(), pdf)

    def test_already_present_is_noop(self, world):
        _published(world)
        n_runs = len(world.store.runs(1))
        report = add(world.deps(), os.path.join(world.pdf_dir, "10_1000_a.pdf"))
        assert report.status == AddStatus.ALREADY_PRESENT
        assert len(world.store.runs(1)) == n_runs

    def test_force_makes_new_run_active(self, world):
        snap = _published(world)
        gid = world.store.get_guideline(_A).id
        old = world.store.active_run(snap.id, gid).id

        report = add(world.deps(), os.path.join(world.pdf_dir, "10_1000_a.pdf"), IngestMode.FORCE)

        assert report.status == AddStatus.ADDED
        assert world.store.active_run(snap.id, gid).id != old

    def test_new_guideline_from_elsewhere(self, world, tmp_path):
        snap = _published(world)
        world.add_row(_D, write_pdf=False)
        outside = tmp_path / "downloads"
        outside.mkdir()
        pdf = world.write_pdf(_D, str(outside))

        report = add(world.deps(), pdf)

        gid = world.store.get_guideline(_D).id
        assert report.status == AddStatus.ADDED
        assert os.path.isfile(os.path.join(world.pdf_dir, "10_1000_d.pdf"))
        assert world.store.members(snap.id)[-1].origin == MemberOrigin.POST_PUBLISH
        assert world.store.active_run(snap.id, gid).status == RunStatus.SUCCEEDED

    def test_duplicate_pdf_bytes(self, world):
        _published(world)
        world.add_row(_D, write_pdf=False)
        a_bytes = open(os.path.join(world.pdf_dir, "10_1000_a.pdf"), "rb").read()
        pdf = os.path.join(world.pdf_dir, "10_1000_d.pdf")
        open(pdf, "wb").write(a_bytes)
        with pytest.raises(DuplicatePdfError):
            add(world.deps(), pdf)


class TestStatus:
    def test_no_snapshot(self, world):
        assert status(world.deps()).snapshot_id is None

    def test_eta_and_failures(self, world):
        snapshot(world.deps(runner=FakeRunner(fail=(_C,))), _NEW, CONFIG)

        report = status(world.deps())

        assert (report.n_succeeded, report.n_failed, report.pending) == (2, 1, [_C])
        assert report.eta_s is not None and report.eta_s >= 0
        assert report.failures[0][0] == _C

    def test_slow_flag(self, world):
        snapshot(world.deps(), _NEW, CONFIG)
        gid = world.store.get_guideline(_C).id
        # Make C's run 10× the others
        world.store._conn.execute("UPDATE extraction_run SET duration_s = 1.0")
        world.store._conn.execute("UPDATE extraction_run SET duration_s = 10.0 WHERE guideline_id = ?", (gid,))
        assert [doi for doi, _ in status(world.deps()).slow] == [_C]

    def test_low_recall_outlier(self, world):
        # B (ESICM) extracts 9 recs, C (ESICM) 1: below half of the ESICM median 5
        class _Verbose(FakeRunner):
            def __call__(self, cfg, pdf_path, doi):
                out = super().__call__(cfg, pdf_path, doi)
                return dataclasses.replace(out, recs=out.recs * 3) if doi == _B else out
        snapshot(world.deps(runner=_Verbose()), _NEW, CONFIG)
        outliers = status(world.deps()).outliers
        assert [doi for doi, _ in outliers] == [_C]

    def test_coverage_gap(self, world):
        snapshot(world.deps(), _NEW, CONFIG)
        world.add_row(_D)
        assert status(world.deps()).coverage_gaps == [_D]


class TestModelLabels:
    def test_reports_name_the_model(self, world):
        _published(world)
        snap = snapshot(world.deps(), _NEW, WORSE_CONFIG)
        val = validate(world.deps())

        assert snap.model == "qwen3:8b@digest"
        assert (val.model, val.baseline_model) == ("qwen3:8b@digest", "qwen3:14b@digest")
        assert status(world.deps()).model == "qwen3:8b@digest"

    def test_headline_names_the_model(self, world):
        _published(world)
        rows = list(csv.DictReader(open(world.headline)))
        assert {r["model"] for r in rows} == {"qwen3:14b@digest"}


class TestChooseAndReject:
    """Several COMPLETE candidates: publish must target one explicitly; rejected ones are inert."""

    def _two_candidates(self, world):
        _published(world)
        first = snapshot(world.deps(), _NEW, CONFIG).snapshot_id
        validate(world.deps(), first)
        second = snapshot(world.deps(), _NEW, CONFIG).snapshot_id
        validate(world.deps(), second)
        return first, second

    def test_publish_without_id_refuses_ambiguity(self, world):
        first, second = self._two_candidates(world)
        with pytest.raises(AmbiguousPublishError) as err:
            publish(world.deps())
        assert err.value.snapshot_ids == [first, second]

    def test_publish_chosen_snapshot(self, world):
        first, second = self._two_candidates(world)

        report = publish(world.deps(), snapshot_id=first)

        assert report.snapshot_id == first and not report.republished
        assert world.store.get_snapshot(first).state == SnapshotState.PUBLISHED
        assert world.store.get_snapshot(second).state == SnapshotState.COMPLETE

    def test_stale_validation_refused(self, world):
        first, second = self._two_candidates(world)
        publish(world.deps(), snapshot_id=first)

        # second was gated against snapshot 1, no longer the published baseline
        with pytest.raises(StaleValidationError):
            publish(world.deps(), snapshot_id=second)
        validate(world.deps(), second)
        assert publish(world.deps(), snapshot_id=second).snapshot_id == second

    def test_rejected_snapshot_is_ignored(self, world):
        first, second = self._two_candidates(world)

        reject(world.deps(), second, "thinking on: worse F1")

        snap = world.store.get_snapshot(second)
        assert (snap.state, snap.reject_reason) == (SnapshotState.REJECTED, "thinking on: worse F1")
        assert status(world.deps()).snapshot_id == first
        assert publish(world.deps()).snapshot_id == first
        with pytest.raises(NotPublishableError):
            publish(world.deps(), snapshot_id=second)

    def test_reject_building_unblocks_new(self, world):
        building = snapshot(world.deps(runner=FakeRunner(fail=(_A,))), _NEW, CONFIG).snapshot_id
        reject(world.deps(), building, "abandoned")
        assert snapshot(world.deps(), _NEW, CONFIG).state == SnapshotState.COMPLETE

    def test_reject_published_refused(self, world):
        published = _published(world)
        with pytest.raises(InvalidSnapshotTransitionError):
            reject(world.deps(), published.id, "no")

    def test_reject_needs_reason(self, world):
        snap = snapshot(world.deps(), _NEW, CONFIG).snapshot_id
        with pytest.raises(ValueError):
            reject(world.deps(), snap, "  ")

    def test_older_published_not_republishable(self, world):
        first, _ = self._two_candidates(world)
        publish(world.deps(), snapshot_id=first)
        with pytest.raises(NotPublishableError):
            publish(world.deps(), snapshot_id=1)

    def test_republish_by_id_with_candidate_pending(self, world):
        # What `add` relies on: re-publishing the published snapshot ignores a COMPLETE candidate
        published = _published(world)
        candidate = snapshot(world.deps(), _NEW, CONFIG).snapshot_id
        validate(world.deps(), candidate)

        report = publish(world.deps(), snapshot_id=published.id)

        assert report.republished and report.snapshot_id == published.id
        assert world.store.get_snapshot(candidate).state == SnapshotState.COMPLETE
