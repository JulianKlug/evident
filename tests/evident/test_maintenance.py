"""M5 maintenance mode: one-guideline adds, publication gates, live duplicate ledger, release record, rebind-pdf.

    World (a, b, c published, live ledger seeded) ─▶ manifest + candidate row ─▶ add ─▶ gates ─▶ outputs + release.json

Every gate failure must leave the previous public outputs byte-identical (spec M5 §5).
"""

import csv
import dataclasses
import json
import os

import pytest

from evident.corpus import file_sha256
from evident.dedup import DedupVerdictError, Verdict, merge_rows, read_verdict_rows, write_adjudications
from evident.domain import SnapshotState
from evident.extraction import ArtifactMismatchError, Artifacts
from evident.pipeline import (
    AddStatus,
    CoverageError,
    GateFailedError,
    IngestMode,
    NoActiveRunError,
    PdfChangedError,
    PromptMismatchError,
    PublishBlockedError,
    RebindStatus,
    SnapshotStart,
    StatusCheck,
    StatusTarget,
    UnresolvedDuplicatesError,
    add,
    publish,
    rebind_pdf,
    reject,
    snapshot,
    status,
    validate,
)
from evident.release import RELEASE_FILE, ReleaseOutcome
from evident.store import DuplicatePdfError, Store
from tests.evident.pipeline_fakes import (
    CONFIG,
    EXTRA_PAGE,
    GUIDELINES,
    META_ONLY,
    TEXT_CHANGE,
    WORSE_CONFIG,
    FakeRunner,
    LiveWorld,
)
from utils.doi import doi_to_filename

_A, _B, _C, _D, _F = "10.1000/a", "10.1000/b", "10.1000/c", "10.1000/d", "10.1000/f"
_F_COPY = (_F, 0, 2)  # f row 2 restates row 0 with its grade appended


@pytest.fixture
def world(tmp_path):
    w = LiveWorld(tmp_path)
    snapshot(w.live(), SnapshotStart.NEW, CONFIG)
    validate(w.live())
    w.judge(1)
    publish(w.live())
    yield w
    w.store.close()


def _active_run_ids(world, snapshot_id=1) -> dict[str, int]:
    return {g.doi: world.store.active_run(snapshot_id, g.id).id for g in world.store.list_guidelines()
            if world.store.active_run(snapshot_id, g.id)}


def _headline_graded(world, society: str) -> str:
    with open(world.headline, newline="") as f:
        return {r["society"]: r for r in csv.DictReader(f)}[society]["n_graded"]


# ── 1–2: incremental add ────────────────────────────────────────

class TestIncrementalAdd:
    def test_extracts_only_the_new_doi_with_the_published_version(self, world):
        before = _active_run_ids(world)
        world.runner.calls.clear()
        world.add_guideline(_D)

        report = add(world.live(), world.pdf(_D))

        assert (report.status, report.snapshot_id) == (AddStatus.ADDED, 1)
        assert world.runner.calls == [_D]
        after = _active_run_ids(world)
        assert {doi: after[doi] for doi in before} == before
        assert world.store.latest_snapshot(SnapshotState.PUBLISHED).id == 1

    def test_same_pdf_twice_is_idempotent(self, world):
        world.add_guideline(_D)
        add(world.live(), world.pdf(_D))
        world.runner.calls.clear()

        assert add(world.live(), world.pdf(_D)).status == AddStatus.ALREADY_PRESENT
        assert world.runner.calls == []

    def test_changed_pdf_needs_force_and_keeps_the_recorded_bytes(self, world, tmp_path):
        recorded = file_sha256(world.pdf(_A))
        changed = world.append_to_pdf(_A, META_ONLY, directory=str(tmp_path / "downloads"))

        with pytest.raises(PdfChangedError):
            add(world.live(), changed)
        assert file_sha256(world.pdf(_A)) == recorded

        assert add(world.live(), changed, IngestMode.FORCE).status == AddStatus.ADDED

    def test_candidate_parity_is_checked_before_extraction(self, world):
        world.add_row(_D)  # manifest row, no included candidate
        world.runner.calls.clear()

        with pytest.raises(PublishBlockedError, match="not an included candidate"):
            add(world.live(), world.pdf(_D))
        assert world.runner.calls == []


# ── 3: gates on re-publish ──────────────────────────────────────

class TestRepublishGates:
    def test_new_manifest_row_without_a_run(self, world):
        before = world.output_hashes()
        world.add_guideline(_D)

        with pytest.raises(CoverageError):
            publish(world.live())
        assert world.output_hashes() == before

    @pytest.mark.parametrize("damage", ["changed", "missing"])
    def test_pdf_changed_or_missing(self, world, damage):
        before = world.output_hashes()
        if damage == "changed":
            world.append_to_pdf(_A, META_ONLY)
        else:
            os.remove(world.pdf(_A))

        with pytest.raises(PublishBlockedError, match=_A):
            publish(world.live())
        assert world.output_hashes() == before

    def test_candidate_without_manifest_row(self, world):
        before = world.output_hashes()
        world.write_candidates(extra=(_D,))

        with pytest.raises(PublishBlockedError, match="has no manifest row"):
            publish(world.live())
        assert world.output_hashes() == before

    def test_search_window_missing(self, world):
        os.remove(world.search)

        with pytest.raises(PublishBlockedError, match="search window"):
            publish(world.live())

    def test_extractor_artefact_drift(self, world):
        world.probe.artifacts = Artifacts("re-pulled digest", None, "rev", "pool")

        with pytest.raises(PublishBlockedError, match="model_digest"):
            publish(world.live())


# ── 4: live duplicate ledger ────────────────────────────────────

class TestDuplicateLedger:
    def test_new_pairs_block_until_judged(self, world):
        world.add_guideline(_F)
        add(world.live(), world.pdf(_F))
        before = world.output_hashes()

        with pytest.raises(UnresolvedDuplicatesError) as err:
            publish(world.live())
        assert [(p.doi, p.ordinal_a, p.ordinal_b) for p in err.value.pairs] == [_F_COPY]
        assert world.output_hashes() == before

        world.judge(1, duplicates=(_F_COPY,))
        report = publish(world.live())
        assert report.n_duplicates_removed == 1

    def test_verdict_for_replaced_text_is_not_applied(self, world):
        world.add_guideline(_F)
        add(world.live(), world.pdf(_F))
        world.judge(1, duplicates=(_F_COPY,))
        publish(world.live())

        # Re-extraction: ordinal 2 is now another statement; the old verdict must not merge it
        rows = GUIDELINES[_F][3]
        reworded = [rows[0], rows[1], ("We suggest prone positioning in severe ARDS for 16 hours", *rows[2][1:])]
        runner = FakeRunner()
        GUIDELINES[_F] = (*GUIDELINES[_F][:3], reworded)
        try:
            add(world.live(runner), world.pdf(_F), IngestMode.FORCE)
            with pytest.raises(UnresolvedDuplicatesError):
                publish(world.live())
        finally:
            GUIDELINES[_F] = (*GUIDELINES[_F][:3], rows)

    def test_conflicting_verdicts_for_the_same_texts(self, world):
        world.add_guideline(_F)
        add(world.live(), world.pdf(_F))
        world.judge(1, duplicates=(_F_COPY,))
        (pair,) = [r[0] for r in read_verdict_rows(world.ledger) if (r[0].doi, r[0].ordinal_a) == _F_COPY[:2]]

        with pytest.raises(DedupVerdictError):
            merge_rows(read_verdict_rows(world.ledger), [(pair, Verdict.DISTINCT, "r", "m", "d")])

        rows = read_verdict_rows(world.ledger) + [(pair, Verdict.DISTINCT, "r", "m", "d")]
        write_adjudications(world.ledger, rows)  # hand-edited ledger
        with pytest.raises(DedupVerdictError):
            publish(world.live())


# ── 5: the M4 freeze is untouched ───────────────────────────────

class TestFreeze:
    def test_live_publish_never_writes_the_frozen_verdicts(self, world, tmp_path):
        frozen = tmp_path / "paper1_verdicts.csv"
        write_adjudications(str(frozen), read_verdict_rows(world.ledger))
        frozen_sha = file_sha256(str(frozen))

        world.add_guideline(_F)
        add(world.live(), world.pdf(_F))
        world.judge(1, duplicates=(_F_COPY,))
        publish(world.live())

        assert file_sha256(str(frozen)) == frozen_sha

    def test_repo_freeze_pin_and_live_default(self):
        with open("configs/paper1.json") as f:
            cfg = json.load(f)
        assert file_sha256(cfg["dedup_verdicts_path"]) == cfg["dedup_verdicts_sha256"]

        import evident.cli as cli
        assert cli._DEFAULT_DEDUP_VERDICTS != cfg["dedup_verdicts_path"]
        live = {(r[0].doi, r[0].ordinal_a, r[0].ordinal_b, r[0].text_a, r[0].text_b, r[1])
                for r in read_verdict_rows(cli._DEFAULT_DEDUP_VERDICTS)}
        frozen = {(r[0].doi, r[0].ordinal_a, r[0].ordinal_b, r[0].text_a, r[0].text_b, r[1])
                  for r in read_verdict_rows(cfg["dedup_verdicts_path"])}
        assert frozen <= live  # seeded from the M4 decisions


# ── 6: editions ─────────────────────────────────────────────────

class TestEditions:
    def test_whole_replacement_moves_the_current_count_not_trends(self, world):
        world.add_guideline(_F, supersedes=_C, topic="topic-c")
        add(world.live(), world.pdf(_F))
        world.judge(1, duplicates=(_F_COPY,))

        report = publish(world.live())

        assert (report.n_headline_guidelines, report.n_skipped_superseded) == (3, 1)  # a, b, f current; c history
        with open(os.path.join(world.publish_dir, "tables", "trends.csv")) as f:
            trends = f.read()
        assert "2025" in trends and "2026" in trends

    def test_focused_update_stays_current_beside_its_parent(self, world):
        world.add_guideline(_F, topic="topic-c-focused")
        add(world.live(), world.pdf(_F))
        world.judge(1, duplicates=(_F_COPY,))

        report = publish(world.live())

        assert (report.n_headline_guidelines, report.n_skipped_superseded) == (4, 0)


# ── 7: strict status ────────────────────────────────────────────

class TestStatus:
    def test_published_target_while_a_candidate_builds(self, world):
        snapshot(world.live(FakeRunner(fail=(_A,))), SnapshotStart.NEW, WORSE_CONFIG)

        assert status(world.live()).snapshot_id == 2
        report = status(world.live(), StatusTarget.PUBLISHED, StatusCheck.ON)
        assert (report.snapshot_id, report.blockers) == (1, [])
        assert report.release_id is not None

    def test_blockers_fail_the_check(self, world):
        world.append_to_pdf(_A, META_ONLY)

        report = status(world.live(), StatusTarget.PUBLISHED, StatusCheck.ON)

        assert any(_A in b for b in report.blockers)

    def test_unresolved_pairs_and_stale_outputs(self, world):
        world.add_guideline(_F)
        add(world.live(), world.pdf(_F))  # staged: not yet published

        report = status(world.live(), StatusTarget.PUBLISHED, StatusCheck.ON)

        assert report.n_unresolved_pairs == 1
        assert any("predate" in b for b in report.blockers)

    def test_slow_runs_are_warnings_not_blockers(self, world):
        world.store._conn.execute("UPDATE extraction_run SET duration_s = 1.0")
        world.store._conn.execute("UPDATE extraction_run SET duration_s = 10.0 WHERE id = 1")

        report = status(world.live(), StatusTarget.PUBLISHED, StatusCheck.ON)

        assert report.slow and report.blockers == []

    def test_no_check_no_blockers_computed(self, world):
        world.append_to_pdf(_A, META_ONLY)
        assert status(world.live()).blockers == []


# ── 8: extractor upgrade ────────────────────────────────────────

class TestUpgrade:
    def test_failed_gate_keeps_the_published_release(self, world):
        before = world.output_hashes()
        snapshot(world.live(), SnapshotStart.NEW, WORSE_CONFIG)
        validate(world.live())

        with pytest.raises(GateFailedError):
            publish(world.live(), snapshot_id=2)
        assert world.output_hashes() == before
        reject(world.live(), 2, "worse recall")
        assert world.store.latest_snapshot(SnapshotState.PUBLISHED).id == 1

    def test_unjudged_candidate_pairs_block_the_switch(self, world):
        world.add_guideline(_F)
        snapshot(world.live(), SnapshotStart.NEW, dataclasses.replace(CONFIG, seed=1))
        validate(world.live())
        before = world.output_hashes()

        with pytest.raises(UnresolvedDuplicatesError):
            publish(world.live(), snapshot_id=2)
        assert world.store.get_snapshot(2).state == SnapshotState.COMPLETE
        assert world.output_hashes() == before

        world.judge(2, duplicates=(_F_COPY,))
        assert publish(world.live(), snapshot_id=2).republished is False
        assert world.store.latest_snapshot(SnapshotState.PUBLISHED).id == 2


# ── 9: release record ───────────────────────────────────────────

class TestRelease:
    def test_repeated_publish_is_a_no_op(self, world):
        first = world.output_hashes()
        record = json.load(open(os.path.join(world.publish_dir, RELEASE_FILE)))

        report = publish(world.live())

        assert report.release_outcome == ReleaseOutcome.UNCHANGED
        assert report.release_id == record["release_id"]
        assert world.output_hashes() == first  # release.json included: not rewritten

    def test_record_content_and_archive(self, world):
        record = json.load(open(os.path.join(world.publish_dir, RELEASE_FILE)))

        assert record["schema_version"] == 1
        assert record["snapshot"]["id"] == 1
        assert record["search"] == {"start": "2016-01-01", "cutoff": "2026-10-04"}
        assert set(record["inputs"]["pdfs"]) == {_A, _B, _C}
        assert record["inputs"]["dedup_verdicts_sha256"] == file_sha256(world.ledger)
        assert record["counts"]["current"] == 3
        assert "tables/headline.csv" in record["outputs"]

        archived = os.path.join(world.archive, record["release_id"])
        assert json.load(open(os.path.join(archived, RELEASE_FILE))) == record
        with Store.open(os.path.join(archived, "evident.db")) as copy:
            assert copy.latest_snapshot(SnapshotState.PUBLISHED).id == 1

    def test_add_makes_a_new_release_naming_the_previous(self, world):
        first = json.load(open(os.path.join(world.publish_dir, RELEASE_FILE)))["release_id"]
        world.add_guideline(_D)
        add(world.live(), world.pdf(_D))

        report = publish(world.live())

        record = json.load(open(os.path.join(world.publish_dir, RELEASE_FILE)))
        assert report.release_outcome == ReleaseOutcome.NEW
        assert (record["release_id"], record["previous_release_id"]) == (report.release_id, first)
        assert record["counts"]["current"] == 4

    def test_render_failure_keeps_the_previous_release(self, world, monkeypatch):
        import evident.pipeline as pipeline
        world.add_guideline(_D)
        add(world.live(), world.pdf(_D))
        before = world.output_hashes()

        def broken(bundle, out_dir):
            raise RuntimeError("disk full")
        monkeypatch.setattr(pipeline, "stage", broken)

        with pytest.raises(RuntimeError):
            publish(world.live())
        assert world.output_hashes() == before

    def test_dashboard_names_cutoff_and_edition_rule(self, world):
        with open(os.path.join(world.publish_dir, "dashboard", "index.html")) as f:
            html = f.read()
        assert "2026-10-04" in html and "supersedes" in html


# ── 2b: rebind-pdf ──────────────────────────────────────────────

def _never_called(n):
    raise AssertionError("rebind-pdf must not call the model")


class TestRebindPdf:
    def test_metadata_only_change_rebinds_without_the_model(self, world):
        run_before = _active_run_ids(world)[_A]
        recs_before = world.store.recommendations(run_before)
        world.append_to_pdf(_A, META_ONLY)
        new_sha = file_sha256(world.pdf(_A))

        report = rebind_pdf(world.live(FakeRunner(on_call=_never_called)), world.pdf(_A))

        assert (report.status, report.new_sha256) == (RebindStatus.REBOUND, new_sha)
        run = world.store.active_run(1, world.store.get_guideline(_A).id)
        assert (run.id, run.pdf_sha256) == (run_before, new_sha)
        assert world.store.recommendations(run.id) == recs_before
        with open(world.rebind_log) as f:
            assert new_sha in f.read()

    def test_copies_a_downloaded_pdf_into_the_pdf_dir(self, world, tmp_path):
        downloaded = world.append_to_pdf(_A, META_ONLY, directory=str(tmp_path / "downloads"))

        rebind_pdf(world.live(), downloaded)

        assert file_sha256(world.pdf(_A)) == file_sha256(downloaded)

    @pytest.mark.parametrize("marker", [TEXT_CHANGE, EXTRA_PAGE])
    def test_changed_model_input_is_refused(self, world, tmp_path, marker):
        recorded = file_sha256(world.pdf(_A))
        downloaded = world.append_to_pdf(_A, marker, directory=str(tmp_path / "downloads"))

        with pytest.raises(PromptMismatchError, match="chunk"):
            rebind_pdf(world.live(), downloaded)
        assert file_sha256(world.pdf(_A)) == recorded
        run = world.store.active_run(1, world.store.get_guideline(_A).id)
        assert run.pdf_sha256 == recorded

    def test_no_active_run(self, world):
        world.add_guideline(_D)

        with pytest.raises(NoActiveRunError):
            rebind_pdf(world.live(), world.pdf(_D))

    def test_bytes_of_another_doi(self, world, tmp_path):
        os.makedirs(tmp_path / "downloads")
        clash = str(tmp_path / "downloads" / doi_to_filename(_A))
        with open(world.pdf(_B), "rb") as src, open(clash, "wb") as dst:
            dst.write(src.read())

        with pytest.raises(DuplicatePdfError):
            rebind_pdf(world.live(), clash)

    def test_unchanged_bytes(self, world):
        assert rebind_pdf(world.live(), world.pdf(_A)).status == RebindStatus.UNCHANGED


# ── review findings ─────────────────────────────────────────────

class TestReviewFindings:
    def test_status_check_detects_outputs_not_matching_the_record(self, world):
        # e.g. publish --package swapped other tables in, or a crash between render and release write
        with open(os.path.join(world.publish_dir, "tables", "headline.csv"), "a") as f:
            f.write("tampered\n")

        report = status(world.live(), StatusTarget.PUBLISHED, StatusCheck.ON)

        assert any("tables/headline.csv" in b for b in report.blockers)

    def test_release_write_failure_is_caught_by_the_check(self, world, monkeypatch):
        import evident.pipeline as pipeline
        world.add_guideline(_D)
        add(world.live(), world.pdf(_D))

        def broken(*args):
            raise OSError("archive dir not writable")
        monkeypatch.setattr(pipeline, "write_release", broken)
        with pytest.raises(OSError):
            publish(world.live())
        monkeypatch.undo()

        assert status(world.live(), StatusTarget.PUBLISHED, StatusCheck.ON).blockers

    def test_add_hash_clash_keeps_the_recorded_pdf(self, world, tmp_path):
        recorded = file_sha256(world.pdf(_A))
        os.makedirs(tmp_path / "downloads")
        clash = str(tmp_path / "downloads" / doi_to_filename(_A))
        with open(world.pdf(_B), "rb") as src, open(clash, "wb") as dst:
            dst.write(src.read())

        with pytest.raises(DuplicatePdfError):
            add(world.live(), clash, IngestMode.FORCE)
        assert file_sha256(world.pdf(_A)) == recorded

    def test_add_artefact_drift_refused_before_any_write(self, world):
        world.add_guideline(_D, write_pdf=False)
        downloaded = world.write_pdf(_D, directory=world.root)
        world.probe.artifacts = Artifacts("re-pulled digest", None, "rev", "pool")

        with pytest.raises(ArtifactMismatchError):
            add(world.live(), downloaded)
        assert not os.path.exists(world.pdf(_D))

    def test_rebind_with_the_recorded_bytes_restores_them(self, world, tmp_path):
        backup = str(tmp_path / "backup" / doi_to_filename(_A))
        os.makedirs(os.path.dirname(backup))
        with open(world.pdf(_A), "rb") as src, open(backup, "wb") as dst:
            dst.write(src.read())
        world.append_to_pdf(_A, META_ONLY)  # the PDF dir copy drifted

        assert rebind_pdf(world.live(), backup).status == RebindStatus.RESTORED
        assert file_sha256(world.pdf(_A)) == file_sha256(backup)
        assert status(world.live(), StatusTarget.PUBLISHED, StatusCheck.ON).blockers == []


# ── PR #4 review ────────────────────────────────────────────────

def _check(world) -> list[str]:
    return status(world.live(), StatusTarget.PUBLISHED, StatusCheck.ON).blockers


def _record(world) -> str:
    with open(os.path.join(world.publish_dir, RELEASE_FILE)) as f:
        return f.read()


def _fail_once_on(path_suffix: str):
    """os.rename stand-in that fails the first move onto `path_suffix`; the rollback's moves succeed."""
    rename = os.rename
    failed = []

    def fake(src, dst):
        if dst.endswith(path_suffix) and not failed:
            failed.append(dst)
            raise OSError(f"disk full: {dst}")
        rename(src, dst)
    return fake


class TestStaleReleaseInputs:
    def test_new_search_window_is_stale_until_published(self, world):
        with open(world.search, "w") as f:
            json.dump({"start": "2016-01-01", "cutoff": "2026-10-20"}, f)

        assert any("search" in b for b in _check(world))
        publish(world.live())
        assert _check(world) == []

    def test_rebound_pdf_is_stale_until_published(self, world):
        world.append_to_pdf(_A, META_ONLY)
        rebind_pdf(world.live(), world.pdf(_A))

        assert any(_A in b for b in _check(world))
        publish(world.live())
        assert _check(world) == []

    def test_new_code_commit_alone_is_not_stale(self, world):
        live = dataclasses.replace(world.live(), code_sha=lambda: "sha2")

        assert status(live, StatusTarget.PUBLISHED, StatusCheck.ON).blockers == []


class TestAtomicPublish:
    def test_failed_archive_leaves_outputs_and_record(self, world):
        world.add_guideline(_D)
        add(world.live(), world.pdf(_D))
        outputs, record = world.output_hashes(), _record(world)

        def broken(path):
            raise OSError("archive dir not writable")
        world.store.backup = broken
        with pytest.raises(OSError):
            publish(world.live())

        assert (world.output_hashes(), _record(world)) == (outputs, record)

    @pytest.mark.parametrize("subdir", ["tables", "figures", "dashboard"])
    def test_failed_swap_restores_every_output_dir(self, world, monkeypatch, subdir):
        import evident.publishing as publishing
        world.add_guideline(_D)
        add(world.live(), world.pdf(_D))
        outputs, record = world.output_hashes(), _record(world)
        entries = sorted(os.listdir(world.publish_dir))

        monkeypatch.setattr(publishing.os, "rename", _fail_once_on(os.sep + subdir))
        with pytest.raises(OSError):
            publish(world.live())
        monkeypatch.undo()

        assert (world.output_hashes(), _record(world)) == (outputs, record)
        assert sorted(os.listdir(world.publish_dir)) == entries  # no staging or .old leftovers

    def test_failed_candidate_publish_is_blocked_then_recovers(self, world):
        snapshot(world.live(), SnapshotStart.NEW, dataclasses.replace(CONFIG, seed=1))
        validate(world.live())
        outputs = world.output_hashes()
        backup = world.store.backup

        def broken(path):
            raise OSError("archive dir not writable")
        world.store.backup = broken
        with pytest.raises(OSError):
            publish(world.live())
        world.store.backup = backup

        assert world.output_hashes() == outputs
        assert any("predate" in b for b in _check(world))
        publish(world.live(), snapshot_id=2)
        assert _check(world) == []
