"""Tests for evident.store against throwaway SQLite files."""

import sqlite3

import pytest

from evident.domain import (
    Category,
    ExtractorVersion,
    GradingFamily,
    Guideline,
    LlmCall,
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
           "extraction_run", "recommendation", "llm_call"}
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
    for gid in gids:
        _succeed(store, snapshot_id, gid)
    store.transition_snapshot(snapshot_id, SnapshotState.COMPLETE)
    if state == SnapshotState.PUBLISHED:
        store.transition_snapshot(snapshot_id, SnapshotState.PUBLISHED)


class TestMigrations:
    def test_empty_file_gets_version_1_and_all_tables(self, tmp_path):
        path = str(tmp_path / "db.sqlite")
        Store.open(path).close()

        conn = sqlite3.connect(path)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert tables == _TABLES
        assert conn.execute("SELECT version FROM schema_version").fetchall() == [(1,)]

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
