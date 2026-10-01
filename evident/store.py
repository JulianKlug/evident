"""SQLite repository for the evidence map: schema, migrations, CRUD, snapshot state machine.

    extractor_version ─┐
                       ▼
    guideline ──▶ snapshot_member ◀── snapshot
        │                               │
        └────────▶ extraction_run ◀─────┘
                     │        │
            recommendation  llm_call

Only raw extracted values are stored; harmonization is applied on read.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

from evident.domain import (
    Category,
    ExtractionRun,
    ExtractorVersion,
    GradingFamily,
    Guideline,
    LlmCall,
    RawRecommendation,
    RunStatus,
    Snapshot,
    SnapshotState,
    ThinkingMode,
)

_DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "doi:")

# Legal snapshot transitions: BUILDING → COMPLETE → PUBLISHED
_ALLOWED_TRANSITIONS = {
    SnapshotState.BUILDING: SnapshotState.COMPLETE,
    SnapshotState.COMPLETE: SnapshotState.PUBLISHED,
}

# Ordered migrations: (version, statements). Never edit a released one; append.
_MIGRATIONS = [
    (1, [
        "CREATE TABLE schema_version (version INTEGER NOT NULL)",
        """CREATE TABLE guideline (
            id             INTEGER PRIMARY KEY,
            doi            TEXT NOT NULL UNIQUE,
            society        TEXT NOT NULL,
            year           INTEGER NOT NULL,
            title          TEXT NOT NULL,
            topic_id       TEXT NOT NULL,
            supersedes_doi TEXT,
            grading_family TEXT NOT NULL,
            pdf_path       TEXT,
            pdf_sha256     TEXT UNIQUE,
            added_at       TEXT NOT NULL
        )""",
        """CREATE TABLE extractor_version (
            id                 TEXT PRIMARY KEY,
            config_json        TEXT NOT NULL,
            model_name         TEXT NOT NULL,
            model_digest       TEXT NOT NULL,
            classifier_sha256  TEXT,
            embedding_revision TEXT,
            created_at         TEXT NOT NULL
        )""",
        """CREATE TABLE snapshot (
            id                   INTEGER PRIMARY KEY,
            extractor_version_id TEXT NOT NULL REFERENCES extractor_version(id),
            state                TEXT NOT NULL,
            created_at           TEXT NOT NULL,
            completed_at         TEXT,
            published_at         TEXT,
            accept_reason        TEXT
        )""",
        """CREATE TABLE snapshot_member (
            snapshot_id     INTEGER NOT NULL REFERENCES snapshot(id),
            guideline_id    INTEGER NOT NULL REFERENCES guideline(id),
            excluded_reason TEXT,
            PRIMARY KEY (snapshot_id, guideline_id)
        )""",
        """CREATE TABLE extraction_run (
            id            INTEGER PRIMARY KEY,
            snapshot_id   INTEGER NOT NULL REFERENCES snapshot(id),
            guideline_id  INTEGER NOT NULL REFERENCES guideline(id),
            status        TEXT NOT NULL,
            started_at    TEXT NOT NULL,
            finished_at   TEXT,
            duration_s    REAL,
            n_pages       INTEGER,
            n_llm_calls   INTEGER,
            thinking      INTEGER NOT NULL DEFAULT 0,
            few_shot_json TEXT,
            error         TEXT
        )""",
        "CREATE INDEX ix_run_member ON extraction_run(snapshot_id, guideline_id, status)",
        """CREATE TABLE recommendation (
            id            INTEGER PRIMARY KEY,
            run_id        INTEGER NOT NULL REFERENCES extraction_run(id) ON DELETE CASCADE,
            ordinal       INTEGER NOT NULL,
            text          TEXT NOT NULL,
            raw_strength  TEXT NOT NULL,
            raw_certainty TEXT NOT NULL,
            raw_category  TEXT NOT NULL,
            page          INTEGER,
            chunk_id      INTEGER
        )""",
        """CREATE TABLE llm_call (
            id            INTEGER PRIMARY KEY,
            run_id        INTEGER NOT NULL REFERENCES extraction_run(id) ON DELETE CASCADE,
            sample_idx    INTEGER NOT NULL DEFAULT 0,
            chunk_id      INTEGER NOT NULL,
            seed          INTEGER,
            prompt_sha256 TEXT NOT NULL,
            raw_text      TEXT NOT NULL,
            prompt_tokens INTEGER,
            eval_tokens   INTEGER,
            duration_ms   REAL
        )""",
    ]),
]
_CODE_SCHEMA_VERSION = _MIGRATIONS[-1][0]

_THINKING_TO_DB = {ThinkingMode.OFF: 0, ThinkingMode.ON: 1}
_THINKING_FROM_DB = {v: k for k, v in _THINKING_TO_DB.items()}


class StoreError(Exception):
    """Base class for all store errors."""


class SchemaTooNewError(StoreError):
    def __init__(self, db_version: int, code_version: int):
        super().__init__(f"DB schema v{db_version} is newer than code v{code_version}")
        self.db_version = db_version
        self.code_version = code_version


class DuplicatePdfError(StoreError):
    def __init__(self, doi_existing: str, doi_new: str):
        super().__init__(f"PDF of {doi_new} is already stored for {doi_existing}")
        self.doi_existing = doi_existing
        self.doi_new = doi_new


class ExtractorVersionConflictError(StoreError):
    def __init__(self, version_id: str):
        super().__init__(f"Extractor version {version_id} already registered with a different config")
        self.version_id = version_id


class InvalidSnapshotTransitionError(StoreError):
    def __init__(self, from_state: SnapshotState, to_state: Optional[SnapshotState]):
        target = to_state.value if to_state else "member change"
        super().__init__(f"Snapshot cannot go from {from_state.value} to {target}")
        self.from_state = from_state
        self.to_state = to_state


class SnapshotIncompleteError(StoreError):
    def __init__(self, pending: list[int]):
        super().__init__(f"Snapshot has members without a succeeded run: {pending}")
        self.pending = pending


class NotFoundError(StoreError):
    """Referenced snapshot, member or run does not exist."""


def _normalize_doi(doi: str) -> str:
    """'https://doi.org/10.1/ABC ' → '10.1/abc'."""
    doi = doi.strip().lower()
    for prefix in _DOI_PREFIXES:
        if doi.startswith(prefix):
            return doi[len(prefix):]
    return doi


def _now() -> str:
    # Microseconds keep finished_at ordering strict between fast consecutive runs
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class Store:
    """The only public entry point to the evidence-map database."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    @classmethod
    def open(cls, path: str) -> "Store":
        # Autocommit mode; transactions are explicit via _transaction()
        conn = sqlite3.connect(path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        store = cls(conn)
        store._migrate()
        return store

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── guidelines ──────────────────────────────────────────────

    def upsert_guideline(self, g: Guideline) -> int:
        doi = _normalize_doi(g.doi)

        # A PDF belongs to exactly one DOI
        if g.pdf_sha256:
            clash = self._conn.execute(
                "SELECT doi FROM guideline WHERE pdf_sha256 = ? AND doi != ?", (g.pdf_sha256, doi),
            ).fetchone()
            if clash:
                raise DuplicatePdfError(clash["doi"], doi)

        with self._transaction():
            self._conn.execute(
                """INSERT INTO guideline (doi, society, year, title, topic_id, supersedes_doi,
                                          grading_family, pdf_path, pdf_sha256, added_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(doi) DO UPDATE SET
                     society = excluded.society, year = excluded.year, title = excluded.title,
                     topic_id = excluded.topic_id, supersedes_doi = excluded.supersedes_doi,
                     grading_family = excluded.grading_family, pdf_path = excluded.pdf_path,
                     pdf_sha256 = excluded.pdf_sha256""",
                (doi, g.society, g.year, g.title, g.topic_id,
                 _normalize_doi(g.supersedes_doi) if g.supersedes_doi else None,
                 GradingFamily(g.grading_family).value, g.pdf_path, g.pdf_sha256, _now()),
            )
        return self._conn.execute("SELECT id FROM guideline WHERE doi = ?", (doi,)).fetchone()["id"]

    def get_guideline(self, doi: str) -> Optional[Guideline]:
        row = self._conn.execute(
            "SELECT * FROM guideline WHERE doi = ?", (_normalize_doi(doi),),
        ).fetchone()
        return self._guideline_from_row(row) if row else None

    def list_guidelines(self) -> list[Guideline]:
        rows = self._conn.execute("SELECT * FROM guideline ORDER BY id").fetchall()
        return [self._guideline_from_row(r) for r in rows]

    # ── extractor versions ──────────────────────────────────────

    def register_extractor_version(self, v: ExtractorVersion) -> None:
        existing = self.get_extractor_version(v.id)
        if existing:
            if existing.config_json != v.config_json:
                raise ExtractorVersionConflictError(v.id)
            return

        with self._transaction():
            self._conn.execute(
                """INSERT INTO extractor_version (id, config_json, model_name, model_digest,
                                                  classifier_sha256, embedding_revision, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (v.id, v.config_json, v.model_name, v.model_digest,
                 v.classifier_sha256, v.embedding_revision, _now()),
            )

    def get_extractor_version(self, version_id: str) -> Optional[ExtractorVersion]:
        row = self._conn.execute(
            "SELECT * FROM extractor_version WHERE id = ?", (version_id,),
        ).fetchone()
        return ExtractorVersion(**dict(row)) if row else None

    # ── snapshots ───────────────────────────────────────────────

    def create_snapshot(self, version_id: str, guideline_ids: list[int]) -> int:
        with self._transaction():
            cur = self._conn.execute(
                "INSERT INTO snapshot (extractor_version_id, state, created_at) VALUES (?, ?, ?)",
                (version_id, SnapshotState.BUILDING.value, _now()),
            )
            snapshot_id = cur.lastrowid
            self._conn.executemany(
                "INSERT INTO snapshot_member (snapshot_id, guideline_id) VALUES (?, ?)",
                [(snapshot_id, gid) for gid in guideline_ids],
            )
        return snapshot_id

    def add_snapshot_member(self, snapshot_id: int, guideline_id: int) -> None:
        self._require_building(snapshot_id)
        with self._transaction():
            self._conn.execute(
                "INSERT OR IGNORE INTO snapshot_member (snapshot_id, guideline_id) VALUES (?, ?)",
                (snapshot_id, guideline_id),
            )

    def exclude_member(self, snapshot_id: int, guideline_id: int, reason: str) -> None:
        if not reason or not reason.strip():
            raise ValueError("Exclusion reason must be non-empty")

        self._require_building(snapshot_id)
        with self._transaction():
            cur = self._conn.execute(
                "UPDATE snapshot_member SET excluded_reason = ? WHERE snapshot_id = ? AND guideline_id = ?",
                (reason, snapshot_id, guideline_id),
            )
        if cur.rowcount == 0:
            raise NotFoundError(f"Guideline {guideline_id} is not a member of snapshot {snapshot_id}")

    def transition_snapshot(self, snapshot_id: int, to: SnapshotState) -> None:
        current = self._snapshot_state(snapshot_id)
        if _ALLOWED_TRANSITIONS.get(current) != to:
            raise InvalidSnapshotTransitionError(current, to)

        # COMPLETE means every included member has a succeeded run
        if to == SnapshotState.COMPLETE:
            pending = self.pending_members(snapshot_id)
            if pending:
                raise SnapshotIncompleteError(pending)

        timestamp_column = "completed_at" if to == SnapshotState.COMPLETE else "published_at"
        with self._transaction():
            self._conn.execute(
                f"UPDATE snapshot SET state = ?, {timestamp_column} = ? WHERE id = ?",
                (to.value, _now(), snapshot_id),
            )

    def latest_snapshot(self, state: SnapshotState) -> Optional[Snapshot]:
        row = self._conn.execute(
            "SELECT * FROM snapshot WHERE state = ? ORDER BY id DESC LIMIT 1", (state.value,),
        ).fetchone()
        if not row:
            return None
        data = dict(row)
        data["state"] = SnapshotState(data["state"])
        return Snapshot(**data)

    def pending_members(self, snapshot_id: int) -> list[int]:
        rows = self._conn.execute(
            """SELECT m.guideline_id FROM snapshot_member m
               WHERE m.snapshot_id = ? AND m.excluded_reason IS NULL
                 AND NOT EXISTS (SELECT 1 FROM extraction_run r
                                 WHERE r.snapshot_id = m.snapshot_id AND r.guideline_id = m.guideline_id
                                   AND r.status = ?)
               ORDER BY m.guideline_id""",
            (snapshot_id, RunStatus.SUCCEEDED.value),
        ).fetchall()
        return [r["guideline_id"] for r in rows]

    # ── runs ────────────────────────────────────────────────────

    def start_run(self, snapshot_id: int, guideline_id: int, thinking: ThinkingMode,
                  few_shot: list[dict]) -> int:
        with self._transaction():
            cur = self._conn.execute(
                """INSERT INTO extraction_run (snapshot_id, guideline_id, status, started_at,
                                               thinking, few_shot_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (snapshot_id, guideline_id, RunStatus.RUNNING.value, _now(),
                 _THINKING_TO_DB[thinking], json.dumps(few_shot)),
            )
        return cur.lastrowid

    def finish_run(self, run_id: int, recs: list[RawRecommendation],
                   calls: list[LlmCall], n_pages: int) -> RunStatus:
        status = RunStatus.SUCCEEDED if recs else RunStatus.FAILED_EMPTY
        started_at = self._run_started_at(run_id)
        finished_at = _now()
        duration_s = (datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)).total_seconds()

        # One transaction: a crash mid-write leaves the run RUNNING with no rows
        with self._transaction():
            self._insert_recommendations(run_id, recs)
            self._insert_calls(run_id, calls)
            self._conn.execute(
                """UPDATE extraction_run SET status = ?, finished_at = ?, duration_s = ?,
                                              n_pages = ?, n_llm_calls = ?
                   WHERE id = ?""",
                (status.value, finished_at, duration_s, n_pages, len(calls), run_id),
            )
        return status

    def fail_run(self, run_id: int, error: str) -> None:
        self._run_started_at(run_id)
        with self._transaction():
            self._conn.execute(
                "UPDATE extraction_run SET status = ?, finished_at = ?, error = ? WHERE id = ?",
                (RunStatus.FAILED.value, _now(), error, run_id),
            )

    def active_run(self, snapshot_id: int, guideline_id: int) -> Optional[ExtractionRun]:
        # Latest SUCCEEDED run wins; FAILED_EMPTY and stale RUNNING are never active
        row = self._conn.execute(
            """SELECT * FROM extraction_run
               WHERE snapshot_id = ? AND guideline_id = ? AND status = ?
               ORDER BY finished_at DESC, id DESC LIMIT 1""",
            (snapshot_id, guideline_id, RunStatus.SUCCEEDED.value),
        ).fetchone()
        return self._run_from_row(row) if row else None

    def recommendations(self, run_id: int) -> list[RawRecommendation]:
        rows = self._conn.execute(
            """SELECT ordinal, text, raw_strength, raw_certainty, raw_category, page, chunk_id
               FROM recommendation WHERE run_id = ? ORDER BY ordinal""",
            (run_id,),
        ).fetchall()
        return [
            RawRecommendation(**{**dict(r), "raw_category": Category(r["raw_category"])})
            for r in rows
        ]

    # ── private ─────────────────────────────────────────────────

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._conn.execute("BEGIN")
        try:
            yield
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    def _migrate(self) -> None:
        db_version = self._schema_version()
        if db_version > _CODE_SCHEMA_VERSION:
            raise SchemaTooNewError(db_version, _CODE_SCHEMA_VERSION)

        pending = [(v, stmts) for v, stmts in _MIGRATIONS if v > db_version]
        if not pending:
            return

        # All pending migrations in one transaction, in order
        with self._transaction():
            for _, statements in pending:
                for statement in statements:
                    self._conn.execute(statement)
            self._conn.execute("DELETE FROM schema_version")
            self._conn.execute("INSERT INTO schema_version (version) VALUES (?)", (pending[-1][0],))

    def _schema_version(self) -> int:
        has_table = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'",
        ).fetchone()
        if not has_table:
            return 0
        row = self._conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
        return row["v"] or 0

    def _snapshot_state(self, snapshot_id: int) -> SnapshotState:
        row = self._conn.execute("SELECT state FROM snapshot WHERE id = ?", (snapshot_id,)).fetchone()
        if not row:
            raise NotFoundError(f"Snapshot {snapshot_id} does not exist")
        return SnapshotState(row["state"])

    def _require_building(self, snapshot_id: int) -> None:
        state = self._snapshot_state(snapshot_id)
        if state != SnapshotState.BUILDING:
            raise InvalidSnapshotTransitionError(state, None)

    def _run_started_at(self, run_id: int) -> str:
        row = self._conn.execute("SELECT started_at FROM extraction_run WHERE id = ?", (run_id,)).fetchone()
        if not row:
            raise NotFoundError(f"Run {run_id} does not exist")
        return row["started_at"]

    def _insert_recommendations(self, run_id: int, recs: list[RawRecommendation]) -> None:
        self._conn.executemany(
            """INSERT INTO recommendation (run_id, ordinal, text, raw_strength, raw_certainty,
                                           raw_category, page, chunk_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [(run_id, r.ordinal, r.text, r.raw_strength, r.raw_certainty,
              Category(r.raw_category).value, r.page, r.chunk_id) for r in recs],
        )

    def _insert_calls(self, run_id: int, calls: list[LlmCall]) -> None:
        self._conn.executemany(
            """INSERT INTO llm_call (run_id, sample_idx, chunk_id, seed, prompt_sha256, raw_text,
                                     prompt_tokens, eval_tokens, duration_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(run_id, c.sample_idx, c.chunk_id, c.seed, c.prompt_sha256, c.raw_text,
              c.prompt_tokens, c.eval_tokens, c.duration_ms) for c in calls],
        )

    @staticmethod
    def _guideline_from_row(row: sqlite3.Row) -> Guideline:
        data = dict(row)
        data["grading_family"] = GradingFamily(data["grading_family"])
        return Guideline(**data)

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> ExtractionRun:
        data = dict(row)
        data["status"] = RunStatus(data["status"])
        data["thinking"] = _THINKING_FROM_DB[data["thinking"]]
        return ExtractionRun(**data)
