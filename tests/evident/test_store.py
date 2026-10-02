"""Tests for evident.store against throwaway SQLite files."""

import json
import sqlite3

import pytest

import evident.store as store_mod

from evident.domain import (
    Category,
    ExtractorVersion,
    GateResult,
    GradingFamily,
    Guideline,
    LlmCall,
    MemberOrigin,
    RawRecommendation,
    RunStatus,
    SnapshotState,
    ThinkingMode,
)
from evident.store import (
    DuplicatePdfError,
    ExtractorVersionConflictError,
    InvalidSnapshotTransitionError,
    SchemaTooNewError,
    SnapshotIncompleteError,
    Store,
)

_TABLES = {"schema_version", "guideline", "extractor_version", "snapshot", "snapshot_member",
           "extraction_run", "recommendation", "llm_call", "validation"}
_VERSION = ExtractorVersion(id="v1", config_json='{"a": 1}', model_name="qwen3:14b", model_digest="d")
_REC = RawRecommendation(ordinal=0, text="Give X", raw_strength="Strong", raw_certainty="High",
                         raw_category=Category.GRADED, page=3, chunk_id=2)
_CALL = LlmCall(chunk_id=0, prompt_sha256="p", raw_text="Give X | Strong | High", seed=1)


def _guideline(doi="10.1/a", sha=None):
    return Guideline(doi=doi, society="ACP", year=2023, title="T", topic_id="topic",
                     grading_family=GradingFamily.GRADE, pdf_sha256=sha)


@pytest.fixture
def store(tmp_path):
    with Store.open(str(tmp_path / "db.sqlite")) as s:
        yield s


@pytest.fixture
def snapshot(store):
    """A BUILDING snapshot with two members; returns (snapshot_id, [guideline ids])."""
    store.register_extractor_version(_VERSION)
    gids = [store.upsert_guideline(_guideline("10.1/a")), store.upsert_guideline(_guideline("10.1/b"))]
    return store.create_snapshot("v1", gids), gids


def _succeed(store, snapshot_id, gid):
    run = store.start_run(snapshot_id, gid, ThinkingMode.OFF, few_shot=[])
    store.finish_run(run, [_REC], [_CALL], n_pages=10)
    return run


def _to_state(store, snapshot_id, gids, state):
    """Drive a fresh snapshot to `state` along the legal path."""
    if state == SnapshotState.BUILDING:
        return
    if state == SnapshotState.REJECTED:
        store.reject_snapshot(snapshot_id, "test")
        return
    for gid in gids:
        _succeed(store, snapshot_id, gid)
    store.transition_snapshot(snapshot_id, SnapshotState.COMPLETE)
    if state == SnapshotState.PUBLISHED:
        store.transition_snapshot(snapshot_id, SnapshotState.PUBLISHED)


class TestMigrations:
    def test_empty_file_gets_latest_version_and_all_tables(self, tmp_path):
        path = str(tmp_path / "db.sqlite")
        Store.open(path).close()

        conn = sqlite3.connect(path)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert tables == _TABLES
        assert conn.execute("SELECT version FROM schema_version").fetchall() == [(3,)]

    def test_v1_file_migrates_to_latest_keeping_rows(self, tmp_path, monkeypatch):
        path = str(tmp_path / "db.sqlite")

        # Build a v1 file with one snapshot, member and run
        with monkeypatch.context() as m:
            m.setattr(store_mod, "_MIGRATIONS", store_mod._MIGRATIONS[:1])
            m.setattr(store_mod, "_CODE_SCHEMA_VERSION", 1)
            conn = sqlite3.connect(path, isolation_level=None)
            conn.row_factory = sqlite3.Row
            v1 = Store(conn)
            v1._migrate()
            v1.register_extractor_version(_VERSION)
            gid = v1.upsert_guideline(_guideline())
            # Raw SQL: create_snapshot now writes the v2 origin column
            conn.execute("INSERT INTO snapshot (extractor_version_id, state, created_at) VALUES ('v1', 'building', 't')")
            conn.execute("INSERT INTO snapshot_member (snapshot_id, guideline_id) VALUES (1, ?)", (gid,))
            conn.execute("""INSERT INTO extraction_run (snapshot_id, guideline_id, status, started_at)
                            VALUES (1, ?, 'running', 't')""", (gid,))
            conn.close()

        with Store.open(path) as s:
            assert s.members(1) == [store_mod.SnapshotMember(gid, None, MemberOrigin.INITIAL)]
            assert s.runs(1)[0].code_sha is None
            assert s.get_guideline("10.1/a") is not None


    def test_reopen_is_noop(self, tmp_path):
        path = str(tmp_path / "db.sqlite")
        with Store.open(path) as s:
            s.upsert_guideline(_guideline())
        with Store.open(path) as s:
            assert len(s.list_guidelines()) == 1
        assert sqlite3.connect(path).execute("SELECT COUNT(*) FROM schema_version").fetchone() == (1,)

    def test_newer_schema_refused(self, tmp_path):
        path = str(tmp_path / "db.sqlite")
        Store.open(path).close()
        conn = sqlite3.connect(path)
        conn.execute("UPDATE schema_version SET version = 99")
        conn.commit()
        conn.close()

        with pytest.raises(SchemaTooNewError):
            Store.open(path)


class TestGuidelines:
    def test_upsert_inserts_then_updates(self, store):
        gid = store.upsert_guideline(_guideline())
        updated = Guideline(**{**_guideline().__dict__, "title": "New"})

        assert store.upsert_guideline(updated) == gid
        assert store.get_guideline("10.1/a").title == "New"
        assert len(store.list_guidelines()) == 1

    def test_doi_normalized(self, store):
        store.upsert_guideline(_guideline(" https://doi.org/10.7326/M22-2056 "))
        assert store.get_guideline("10.7326/m22-2056").doi == "10.7326/m22-2056"
        assert store.get_guideline("HTTPS://DOI.ORG/10.7326/M22-2056") is not None

    def test_duplicate_pdf_refused(self, store):
        store.upsert_guideline(_guideline("10.1/a", sha="abc"))
        with pytest.raises(DuplicatePdfError) as err:
            store.upsert_guideline(_guideline("10.1/b", sha="abc"))
        assert (err.value.doi_existing, err.value.doi_new) == ("10.1/a", "10.1/b")

    def test_same_pdf_same_doi_ok(self, store):
        store.upsert_guideline(_guideline("10.1/a", sha="abc"))
        store.upsert_guideline(_guideline("10.1/a", sha="abc"))


class TestExtractorVersions:
    def test_register_idempotent(self, store):
        store.register_extractor_version(_VERSION)
        store.register_extractor_version(_VERSION)
        assert store.get_extractor_version("v1").config_json == _VERSION.config_json

    def test_conflict(self, store):
        store.register_extractor_version(_VERSION)
        with pytest.raises(ExtractorVersionConflictError):
            store.register_extractor_version(ExtractorVersion(**{**_VERSION.__dict__, "config_json": "{}"}))


class TestSnapshotLifecycle:
    def test_happy_path(self, store, snapshot):
        snapshot_id, gids = snapshot
        assert store.pending_members(snapshot_id) == gids

        _to_state(store, snapshot_id, gids, SnapshotState.PUBLISHED)

        latest = store.latest_snapshot(SnapshotState.PUBLISHED)
        assert latest.id == snapshot_id
        assert latest.completed_at and latest.published_at
        assert store.latest_snapshot(SnapshotState.BUILDING) is None

    @pytest.mark.parametrize("start", list(SnapshotState))
    @pytest.mark.parametrize("target", list(SnapshotState))
    def test_transition_matrix(self, store, snapshot, start, target):
        snapshot_id, gids = snapshot
        _to_state(store, snapshot_id, gids, start)
        legal = (start, target) in {(SnapshotState.BUILDING, SnapshotState.COMPLETE),
                                    (SnapshotState.COMPLETE, SnapshotState.PUBLISHED)}
        if legal:
            if start == SnapshotState.BUILDING:
                for gid in gids:
                    _succeed(store, snapshot_id, gid)
            store.transition_snapshot(snapshot_id, target)
            return
        with pytest.raises(InvalidSnapshotTransitionError):
            store.transition_snapshot(snapshot_id, target)

    def test_complete_with_pending_member_refused(self, store, snapshot):
        snapshot_id, gids = snapshot
        _succeed(store, snapshot_id, gids[0])
        with pytest.raises(SnapshotIncompleteError) as err:
            store.transition_snapshot(snapshot_id, SnapshotState.COMPLETE)
        assert err.value.pending == [gids[1]]

    def test_excluded_member_does_not_block(self, store, snapshot):
        snapshot_id, gids = snapshot
        _succeed(store, snapshot_id, gids[0])
        store.exclude_member(snapshot_id, gids[1], "PDF unavailable")
        store.transition_snapshot(snapshot_id, SnapshotState.COMPLETE)


class TestMembership:
    def test_add_member_after_complete_refused(self, store, snapshot):
        snapshot_id, gids = snapshot
        _to_state(store, snapshot_id, gids, SnapshotState.COMPLETE)
        new_gid = store.upsert_guideline(_guideline("10.1/c"))
        with pytest.raises(InvalidSnapshotTransitionError):
            store.add_snapshot_member(snapshot_id, new_gid)

    def test_add_member_while_building(self, store, snapshot):
        snapshot_id, gids = snapshot
        new_gid = store.upsert_guideline(_guideline("10.1/c"))
        store.add_snapshot_member(snapshot_id, new_gid)
        assert store.pending_members(snapshot_id) == gids + [new_gid]

    @pytest.mark.parametrize("reason", ["", "   "])
    def test_exclude_requires_reason(self, store, snapshot, reason):
        snapshot_id, gids = snapshot
        with pytest.raises(ValueError):
            store.exclude_member(snapshot_id, gids[0], reason)


class TestActiveRun:
    def test_later_succeeded_run_wins(self, store, snapshot):
        snapshot_id, gids = snapshot
        _succeed(store, snapshot_id, gids[0])
        later = _succeed(store, snapshot_id, gids[0])

        active = store.active_run(snapshot_id, gids[0])
        assert active.id == later
        assert active.status == RunStatus.SUCCEEDED
        assert active.n_llm_calls == 1 and active.n_pages == 10

    def test_failed_empty_and_running_never_active(self, store, snapshot):
        snapshot_id, gids = snapshot
        empty = store.start_run(snapshot_id, gids[0], ThinkingMode.OFF, few_shot=[])
        assert store.finish_run(empty, [], [_CALL], n_pages=10) == RunStatus.FAILED_EMPTY
        store.start_run(snapshot_id, gids[0], ThinkingMode.ON, few_shot=[])

        assert store.active_run(snapshot_id, gids[0]) is None
        assert store.pending_members(snapshot_id) == gids

    def test_recommendations_round_trip(self, store, snapshot):
        snapshot_id, gids = snapshot
        run = _succeed(store, snapshot_id, gids[0])
        assert store.recommendations(run) == [_REC]


class TestAtomicity:
    def test_mid_write_failure_leaves_no_partial_run(self, store, snapshot, monkeypatch):
        snapshot_id, gids = snapshot
        run = store.start_run(snapshot_id, gids[0], ThinkingMode.OFF, few_shot=[{"recommendation": "x"}])

        # Recs are written, then the call insert crashes
        def crash(*args):
            raise RuntimeError("disk full")
        monkeypatch.setattr(store, "_insert_calls", crash)
        with pytest.raises(RuntimeError):
            store.finish_run(run, [_REC], [_CALL], n_pages=10)

        assert store.recommendations(run) == []
        assert store.active_run(snapshot_id, gids[0]) is None
        status = store._conn.execute("SELECT status FROM extraction_run WHERE id = ?", (run,)).fetchone()[0]
        assert status == RunStatus.RUNNING.value


class TestIntegrity:
    def test_foreign_keys_enforced(self, store, snapshot):
        snapshot_id, _ = snapshot
        with pytest.raises(sqlite3.IntegrityError):
            store.start_run(snapshot_id, 999, ThinkingMode.OFF, few_shot=[])


class TestOrigins:
    def test_initial_and_rescan(self, store, snapshot):
        snapshot_id, gids = snapshot
        new_gid = store.upsert_guideline(_guideline("10.1/c"))
        store.add_snapshot_member(snapshot_id, new_gid)

        assert [(m.guideline_id, m.origin) for m in store.members(snapshot_id)] == [
            (gids[0], MemberOrigin.INITIAL), (gids[1], MemberOrigin.INITIAL), (new_gid, MemberOrigin.RESCAN)]

    @pytest.mark.parametrize("state", [SnapshotState.BUILDING, SnapshotState.COMPLETE])
    def test_published_member_refused_before_publish(self, store, snapshot, state):
        snapshot_id, gids = snapshot
        _to_state(store, snapshot_id, gids, state)
        new_gid = store.upsert_guideline(_guideline("10.1/c"))
        with pytest.raises(InvalidSnapshotTransitionError):
            store.add_published_member(snapshot_id, new_gid)

    def test_published_member(self, store, snapshot):
        snapshot_id, gids = snapshot
        _to_state(store, snapshot_id, gids, SnapshotState.PUBLISHED)
        new_gid = store.upsert_guideline(_guideline("10.1/c"))

        store.add_published_member(snapshot_id, new_gid)

        assert store.members(snapshot_id)[-1].origin == MemberOrigin.POST_PUBLISH


class TestPublishAndRuns:
    def test_accept_reason_stored(self, store, snapshot):
        snapshot_id, gids = snapshot
        _to_state(store, snapshot_id, gids, SnapshotState.COMPLETE)
        store.transition_snapshot(snapshot_id, SnapshotState.PUBLISHED, accept_reason="noise on 12 guidelines")
        assert store.get_snapshot(snapshot_id).accept_reason == "noise on 12 guidelines"

    def test_get_snapshot_missing(self, store):
        assert store.get_snapshot(42) is None

    def test_runs_all_statuses_with_code_sha(self, store, snapshot):
        snapshot_id, gids = snapshot
        failed = store.start_run(snapshot_id, gids[0], ThinkingMode.OFF, few_shot=[], code_sha="abc")
        store.fail_run(failed, "boom")
        _succeed(store, snapshot_id, gids[0])

        runs = store.runs(snapshot_id)
        assert [(r.status, r.code_sha) for r in runs] == [(RunStatus.FAILED, "abc"), (RunStatus.SUCCEEDED, None)]

    def test_finish_run_records_few_shot(self, store, snapshot):
        snapshot_id, gids = snapshot
        run = store.start_run(snapshot_id, gids[0], ThinkingMode.OFF, few_shot=[])
        store.finish_run(run, [_REC], [_CALL], n_pages=1, few_shot=[{"recommendation": "x"}])
        assert json.loads(store.active_run(snapshot_id, gids[0]).few_shot_json) == [{"recommendation": "x"}]


class TestValidations:
    def test_round_trip_latest_wins(self, store, snapshot):
        snapshot_id, _ = snapshot
        assert store.latest_validation(snapshot_id) is None

        store.save_validation(snapshot_id, None, GateResult.NO_BASELINE, '{"a": 1}')
        second = store.save_validation(snapshot_id, None, GateResult.PASS, '{"a": 2}')

        latest = store.latest_validation(snapshot_id)
        assert (latest.id, latest.gate, latest.report_json, latest.baseline_snapshot_id) == (
            second, GateResult.PASS, '{"a": 2}', None)


class TestReject:
    @pytest.mark.parametrize("state", [SnapshotState.BUILDING, SnapshotState.COMPLETE])
    def test_reject_with_reason(self, store, snapshot, state):
        snapshot_id, gids = snapshot
        _to_state(store, snapshot_id, gids, state)

        store.reject_snapshot(snapshot_id, "worse F1")

        snap = store.get_snapshot(snapshot_id)
        assert (snap.state, snap.reject_reason) == (SnapshotState.REJECTED, "worse F1")
        assert snap.rejected_at
        assert store.latest_snapshot(state) is None

    def test_reject_published_refused(self, store, snapshot):
        snapshot_id, gids = snapshot
        _to_state(store, snapshot_id, gids, SnapshotState.PUBLISHED)
        with pytest.raises(InvalidSnapshotTransitionError):
            store.reject_snapshot(snapshot_id, "no")

    def test_rejected_is_terminal(self, store, snapshot):
        snapshot_id, _ = snapshot
        store.reject_snapshot(snapshot_id, "abandoned")
        with pytest.raises(InvalidSnapshotTransitionError):
            store.transition_snapshot(snapshot_id, SnapshotState.COMPLETE)
        with pytest.raises(InvalidSnapshotTransitionError):
            store.reject_snapshot(snapshot_id, "again")

    def test_reject_needs_reason(self, store, snapshot):
        snapshot_id, _ = snapshot
        with pytest.raises(ValueError):
            store.reject_snapshot(snapshot_id, " ")
