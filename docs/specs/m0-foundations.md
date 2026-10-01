# Spec: M0 Foundations

Status: DRAFT · 2026-10-01 · Branch: `agent_explor`
Parent docs: `docs/designs/living-evidence-map.md` (decisions), `docs/roadmap.md` (M0)
Tasks: T1 store · T2 harmonization · T3 legacy extractor changes · T6 corpus

## 1. Goal

Build the four foundation layers the pipeline (M1) will sit on: a versioned
results store, a grade harmonizer, a corpus manifest, and the minimal changes
to the legacy extractor (category, page, seed, leak-safe few-shot). Legacy
behaviour must stay identical when the new options are off.

### Non-goals (M0)
- `ExtractorConfig`, `pipeline.py`, CLI, validate, publish, analytics (M1/M3).
- Table-first strategy, FT re-evaluation (M2).
- PDF fetching (T11, M1). It must **not** reuse the Sci-Hub path in
  `extraction/pdf_loader.py:217`.
- Schemes other than GRADE in harmonization (M6).

## 2. Constraints

| Constraint | Consequence |
|------------|-------------|
| Python 3.9 (`python --version` → 3.9.10) | `from __future__ import annotations` in every new module; no `match`, no `X \| Y` at runtime |
| No new dependencies | `sqlite3`, `hashlib`, `csv`, `enum`, `dataclasses` from the stdlib only |
| User rules (global CLAUDE.md) | enums instead of boolean params; private by default (`_name`); early return; constants instead of magic values; one short what/why comment per block; minimal diff in existing files |
| Layering | `corpus → store`; `harmonization` is pure; nothing in `evident/` imports `extraction/` in M0 |
| Bug fixes | test first, see it fail, then fix (ICU loader §6.4) |
| Test style | pytest, mock LLM clients as in `tests/test_extractor.py:12`; new tests under `tests/evident/` |

## 3. Layout

```
evident/
  __init__.py
  domain.py          enums + dataclasses shared by all layers (no logic)
  store.py           SQLite repository: schema, migrations, CRUD, snapshot transitions
  harmonization.py   raw grade + text → HarmonizedGrade (pure)
  corpus.py          manifest load/validate, current editions, sync to store
corpus/
  manifest.csv       header only in M0 (rows added in M3)
tests/evident/
  __init__.py
  test_domain.py  test_store.py  test_harmonization.py  test_corpus.py
tests/fixtures/
  legacy_golden/     recorded LLM responses + expected outputs (§6.0)
```

Dependency graph (arrows = "imports"):
```
 corpus ──▶ store ──▶ domain
   │                    ▲
   └────────────────────┤
 harmonization ─────────┘
```

## 4. Domain (`evident/domain.py`)

All enums are `str`-valued (`class X(str, Enum)`), so they serialize to the DB as-is.

```python
class GradingFamily(str, Enum):      GRADE = "grade"          # M6 adds ABCD_123, ESC_ERS, ACC_AHA
class Strength(str, Enum):           STRONG = "strong"; WEAK = "weak"
class Direction(str, Enum):          FOR = "for"; AGAINST = "against"
class Certainty(str, Enum):          HIGH = "high"; MODERATE = "moderate"; LOW = "low"; VERY_LOW = "very_low"
class Category(str, Enum):           GRADED = "graded"; BEST_PRACTICE = "best_practice"; NO_RECOMMENDATION = "no_recommendation"
class AxisStatus(str, Enum):         MAPPED = "mapped"; UNGRADED = "ungraded"; UNMAPPED = "unmapped"
class RunStatus(str, Enum):          RUNNING = "running"; SUCCEEDED = "succeeded"; FAILED = "failed"; FAILED_EMPTY = "failed_empty"
class SnapshotState(str, Enum):      BUILDING = "building"; COMPLETE = "complete"; PUBLISHED = "published"
```

Dataclasses (frozen): `Guideline`, `ExtractorVersion`, `Snapshot`,
`ExtractionRun`, `RawRecommendation`, `LlmCall`, `HarmonizedGrade`.
Fields match the tables in §5. `HarmonizedGrade` is defined in §7.

## 5. T1 Store (`evident/store.py`)

### 5.1 Schema (migration 1)

```sql
PRAGMA foreign_keys = ON;   -- set on every connection
PRAGMA journal_mode = WAL;  -- publish can read while a snapshot writes

CREATE TABLE schema_version (version INTEGER NOT NULL);

CREATE TABLE guideline (
  id             INTEGER PRIMARY KEY,
  doi            TEXT NOT NULL UNIQUE,          -- lower-cased, no "https://doi.org/"
  society        TEXT NOT NULL,
  year           INTEGER NOT NULL,
  title          TEXT NOT NULL,
  topic_id       TEXT NOT NULL,
  supersedes_doi TEXT,                          -- validated by corpus, not FK (edition may be outside corpus)
  grading_family TEXT NOT NULL,
  pdf_path       TEXT,
  pdf_sha256     TEXT UNIQUE,                   -- NULL until PDF present
  added_at       TEXT NOT NULL                  -- ISO-8601 UTC
);

CREATE TABLE extractor_version (
  id                 TEXT PRIMARY KEY,          -- sha256 hex of canonical config JSON (M1 computes)
  config_json        TEXT NOT NULL,
  model_name         TEXT NOT NULL,
  model_digest       TEXT NOT NULL,
  classifier_sha256  TEXT,
  embedding_revision TEXT,
  created_at         TEXT NOT NULL
);

CREATE TABLE snapshot (
  id                   INTEGER PRIMARY KEY,
  extractor_version_id TEXT NOT NULL REFERENCES extractor_version(id),
  state                TEXT NOT NULL,           -- SnapshotState
  created_at           TEXT NOT NULL,
  completed_at         TEXT,
  published_at         TEXT,
  accept_reason        TEXT                     -- set when published with --accept-regression (M1)
);

CREATE TABLE snapshot_member (
  snapshot_id     INTEGER NOT NULL REFERENCES snapshot(id),
  guideline_id    INTEGER NOT NULL REFERENCES guideline(id),
  excluded_reason TEXT,                         -- NULL = included
  PRIMARY KEY (snapshot_id, guideline_id)
);

CREATE TABLE extraction_run (
  id               INTEGER PRIMARY KEY,
  snapshot_id      INTEGER NOT NULL REFERENCES snapshot(id),
  guideline_id     INTEGER NOT NULL REFERENCES guideline(id),
  status           TEXT NOT NULL,               -- RunStatus
  started_at       TEXT NOT NULL,
  finished_at      TEXT,
  duration_s       REAL,
  n_pages          INTEGER,
  n_llm_calls      INTEGER,
  thinking         INTEGER NOT NULL DEFAULT 0,  -- 0/1 storage only; API uses an enum
  few_shot_json    TEXT,                        -- examples actually used (OV2)
  error            TEXT
);
CREATE INDEX ix_run_member ON extraction_run(snapshot_id, guideline_id, status);

CREATE TABLE recommendation (
  id            INTEGER PRIMARY KEY,
  run_id        INTEGER NOT NULL REFERENCES extraction_run(id) ON DELETE CASCADE,
  ordinal       INTEGER NOT NULL,
  text          TEXT NOT NULL,
  raw_strength  TEXT NOT NULL,                  -- exactly as extracted, before any normalization (OV3)
  raw_certainty TEXT NOT NULL,
  raw_category  TEXT NOT NULL,                  -- Category from parser
  page          INTEGER,                        -- 1-based; NULL only for legacy rows
  chunk_id      INTEGER
);

CREATE TABLE llm_call (
  id            INTEGER PRIMARY KEY,
  run_id        INTEGER NOT NULL REFERENCES extraction_run(id) ON DELETE CASCADE,
  sample_idx    INTEGER NOT NULL DEFAULT 0,
  chunk_id      INTEGER NOT NULL,
  seed          INTEGER,
  prompt_sha256 TEXT NOT NULL,
  raw_text      TEXT NOT NULL,
  prompt_tokens INTEGER, eval_tokens INTEGER, duration_ms REAL
);
```

**Harmonized values are not stored.** `harmonize()` is pure and is applied
on read (M1/M3). A mapping change then needs no migration, and every number can
be recomputed exactly from the stored raw values (OV7).

### 5.2 Public API

`Store` is the only public class. Everything else is private.

```python
class Store:
    @classmethod
    def open(cls, path: str) -> "Store"            # creates file, applies pending migrations
    def close(self) -> None                         # also usable as context manager

    # guidelines
    def upsert_guideline(self, g: Guideline) -> int        # key = doi; refuses sha clash with another doi
    def get_guideline(self, doi: str) -> Guideline | None
    def list_guidelines(self) -> list[Guideline]

    # versions
    def register_extractor_version(self, v: ExtractorVersion) -> None   # idempotent on id; conflict if same id, different config
    def get_extractor_version(self, version_id: str) -> ExtractorVersion | None

    # snapshots
    def create_snapshot(self, version_id: str, guideline_ids: list[int]) -> int   # state=BUILDING, members frozen (OV5)
    def add_snapshot_member(self, snapshot_id: int, guideline_id: int) -> None    # only while BUILDING (re-scan)
    def exclude_member(self, snapshot_id: int, guideline_id: int, reason: str) -> None
    def transition_snapshot(self, snapshot_id: int, to: SnapshotState) -> None
    def latest_snapshot(self, state: SnapshotState) -> Snapshot | None
    def pending_members(self, snapshot_id: int) -> list[int]    # included members without a SUCCEEDED run

    # runs
    def start_run(self, snapshot_id: int, guideline_id: int, thinking: ThinkingMode,
                  few_shot: list[dict]) -> int
    def finish_run(self, run_id: int, recs: list[RawRecommendation],
                   calls: list[LlmCall], n_pages: int) -> RunStatus   # SUCCEEDED, or FAILED_EMPTY if recs == []
    def fail_run(self, run_id: int, error: str) -> None
    def active_run(self, snapshot_id: int, guideline_id: int) -> ExtractionRun | None
    def recommendations(self, run_id: int) -> list[RawRecommendation]
```

`ThinkingMode(str, Enum)`: `OFF`, `ON` (user rule: no boolean params).

### 5.3 Rules

| Rule | Behaviour | Error |
|------|-----------|-------|
| Migrations | `schema_version` read on open; pending migrations applied in one transaction, in order | `SchemaTooNewError` if DB version > code version |
| DOI normalization | lower-case, strip whitespace and `https://doi.org/` prefix | — |
| Duplicate PDF | same `pdf_sha256` already on another DOI | `DuplicatePdfError(doi_existing, doi_new)` |
| Version conflict | same `id`, different `config_json` | `ExtractorVersionConflictError` |
| Snapshot transitions | only `BUILDING→COMPLETE→PUBLISHED`; `COMPLETE` requires `pending_members()==[]` | `InvalidSnapshotTransitionError(from, to)`, `SnapshotIncompleteError(pending)` |
| Member changes | `add_snapshot_member` / `exclude_member` only in `BUILDING` | `InvalidSnapshotTransitionError` |
| `exclude_member` | reason must be non-empty | `ValueError` |
| Active run | latest `SUCCEEDED` run by `finished_at` for (snapshot, guideline); `FAILED_EMPTY` is never active | — |
| `finish_run` atomicity | recs + calls + status written in one transaction (crash = run stays `RUNNING`) | — |
| Stale `RUNNING` | not active; resume (M1) starts a new run | — |

All errors subclass `StoreError`. Each one is a named class in `store.py`.

### 5.4 Tests (`tests/evident/test_store.py`, `tmp_path` DBs)
- **Migrations:**
  - open an empty file → version 1, all tables present;
  - reopening is a no-op;
  - a DB claiming version 99 → `SchemaTooNewError`.
- **Guidelines:**
  - upsert inserts, then updates on the same DOI;
  - DOI normalization;
  - `DuplicatePdfError`.
- **Extractor versions:** idempotent register, and `ExtractorVersionConflictError`.
- **Snapshot, happy path:** create → runs → `COMPLETE` → `PUBLISHED`; `latest_snapshot` returns it.
- **Snapshot, transitions:** every illegal transition raises (parametrized over the 3×3 matrix). `COMPLETE` with a pending member → `SnapshotIncompleteError`; an excluded member doesn't block.
- **Active run:**
  - two `SUCCEEDED` runs → the later one is active;
  - `FAILED_EMPTY` and `RUNNING` are never active.
- **Atomicity:**
  - `finish_run` with an injected failure mid-write leaves no partial recs;
  - an empty rec list gives `FAILED_EMPTY`.
- **Membership:**
  - `add_snapshot_member` after `COMPLETE` → raises;
  - `exclude_member("")` → `ValueError`.
- **Integrity:** foreign keys are enforced (inserting a run for an unknown guideline fails).

## 6. T3 Legacy extractor changes (`extraction/`)

Minimal diff. Every change is off by default or adds a column only.

### 6.0 Golden regression harness (do this FIRST)
Before touching `extraction/`, record the legacy behaviour:
1. `tests/fixtures/legacy_golden/`: 3 PDFs' page texts (one ACP, one ICU, one 3-page synthetic), canned LLM responses per chunk, and the expected `recommendations_df` (3 columns) for:
   - `extract_guideline` (few-shot, `normalize` on and off);
   - `self_consistency_extract` (n=3, fixed fake responses per sample).
2. `tests/test_legacy_golden.py`: replays them through a mock client and asserts the 3 original columns match exactly.

This is the regression gate for §6.1–6.4 and replaces the GPU benchmark (the old
benchmark isn't reproducible: unseeded few-shot, `datasets.py:252`).

### 6.1 Category (1A)
- `prompts.py`:
  - add `class PromptVersion(str, Enum): V0 = "v0"; V1 = "v1"` and the field `PromptStrategy.prompt_version = PromptVersion.V0`;
  - V0 text stays **byte-identical** (test: the prompt for a fixture equals the stored golden string);
  - V1 changes the definition and rules so that ungraded statements are also extracted:
    ```
    - Also extract best-practice / good-practice statements: write BEST_PRACTICE as the {grade_label} and NA as the {level_label}
    - Also extract explicit non-recommendations ("we cannot recommend for or against", "evidence was insufficient to make a recommendation"): write NO_RECOMMENDATION as the {grade_label} and NA as the {level_label}
    ```
    and drops "statements without an explicit grade" from the Do-NOT list.
- `response_parser.py`:
  - the output gains a 4th column, `category`;
  - the grade token `BEST_PRACTICE` → `best_practice`, `NO_RECOMMENDATION` → `no_recommendation`, anything else → `graded`;
  - the 3 original columns are unchanged;
  - empty DataFrames carry all 4 columns.
- `self_consistency.py`: the cluster output carries `category` by majority vote (reuse `_majority_vote`).
- `postprocessing.normalize_extracted_grades`: leaves rows with `category != graded` untouched.
- Tests:
  - parser: graded, best-practice, no-recommendation, mixed, malformed;
  - prompt: V0 golden string; V1 contains both new rules;
  - SC: category vote;
  - golden harness still green.

### 6.2 Page provenance (1D, OV6)
- `extractor.py`:
  - new private `_chunk_spans(pages, pages_per_chunk, overlap) -> list[tuple[int, str]]` returns (1-based start page, text);
  - `_chunk_pages` becomes `[text for _, text in _chunk_spans(...)]` (DRY, same output);
  - in the loop, tag `chunk_df["page"] = start_page` and `chunk_df["chunk_id"] = i` before concat;
  - the empty frame includes both columns.
- `self_consistency.py`: same tagging per sample. The cluster output takes `page = min(cluster pages)` and `chunk_id` of that row.
- `deduplication.py`: when rows collapse (exact or semantic), the kept row's `page` = min across the collapsed rows, applied only if a `page` column exists. Keeping the longer text is unchanged.
- Tests:
  - `_chunk_spans` for `pages_per_chunk` 1 and 3 (with overlap) matches `_chunk_pages` text;
  - dedup keeps the min page;
  - SC keeps the min page;
  - extractor rows carry the correct page for a 3-page fixture;
  - golden harness still green.

### 6.3 Seed (2B)
- `llm_client.py`: `generate`, `_generate_chat` and `generate_json` gain `seed: int | None = None`. If it isn't `None`, `options["seed"] = seed`; otherwise the options dict is byte-identical to today.
- `self_consistency_extract` gains `seed: int | None = None`. Sample *k* calls `generate(..., seed=seed + k)` when a seed is set, and **omits the kwarg** otherwise, so existing mock clients (`tests/test_extractor.py:26`) keep working.
- `extract_guideline` gains `seed: int | None = None`, forwarded only when set.
- Tests:
  - options dict with and without a seed (patch `ollama.Client`);
  - SC passes `seed + k`;
  - no seed → the mock is called without a `seed` kwarg.

### 6.4 Few-shot (OV2) and GT loaders (3A, T6 bug fix)
- `get_few_shot_examples` gains `exclude_doi: str | None = None` and `seed: int | None = None`:
  - exclusion matches `ds.key == exclude_key or ds.doi == exclude_doi` (DOIs normalized as in §5.3);
  - `seed` set → `random.Random(seed).sample(...)`; unset → module `random.sample` (legacy).
- **Bug fix, test first.** `load_icu_datasets` reads only `xlsx_files[0]` (`datasets.py:164`).
  1. Write `test_icu_loader_reads_all_xlsx` (two xlsx in a `tmp_path`, monkeypatch `_ICU_DIR`) and watch it fail.
  2. Fix: read all `*.xlsx` sorted by name, concat, then group by DOI.
  3. Watch it pass. The existing 3 ICU guidelines must load identically (17 rows).
- `GtMode(str, Enum)`: `GRADED_ONLY` (default, today's behaviour), `WITH_UNGRADED`. The ACP and ICU loaders take `gt_mode`.
  - `WITH_UNGRADED` keeps rows whose class is `0`/empty/`No recommendation` and adds `category` via `evident`-independent string rules: class `No recommendation`, or text matching "inconclusive|insufficient|cannot recommend" → `no_recommendation`.
  - Known rows: ACP "Evidence was inconclusive…" (class `0`), ICU "…we cannot recommend…" (class `No recommendation`).
  - Both modes add the columns `raw_class` and `raw_LOE`: the xlsx cell **before** `GRADE.normalize_*`. They are needed because normalization turns "strong recommendation" into "Strong For" even for "recommends against" text; a label-derived "For" would then override the text direction in `harmonize`. Harmonization and M1 scoring read the raw columns only.
- Tests:
  - leakage: for every labelled guideline, no example text equals any of its own GT rows when excluded by DOI;
  - same seed → same examples; different seed → different examples (pool ≥ 10);
  - ICU multi-xlsx;
  - `GtMode` row counts: ACP 29→30, ICU 17 unchanged (the "No recommendation" and bare "recommendation" rows are already kept today); category values as expected; `raw_class`/`raw_LOE` equal the xlsx cells.

## 7. T2 Harmonization (`evident/harmonization.py`)

### 7.1 API
```python
@dataclass(frozen=True)
class HarmonizedGrade:
    category: Category
    strength: Strength | None
    direction: Direction | None
    certainty: Certainty | None
    strength_status: AxisStatus
    certainty_status: AxisStatus

def harmonize(raw_strength: str, raw_certainty: str, text: str,
              raw_category: Category, family: GradingFamily) -> HarmonizedGrade
```
- A non-GRADE family raises `UnsupportedGradingFamilyError` (M6).
- The function is pure and has no I/O.

### 7.2 Rules (in order)
1. **Clean:** lower-case; strip; strip trailing `;,.`; collapse whitespace. Placeholder values `0`, `0.0`, `nan`, `na`, `n/a` and the empty string become empty.
2. **Category:**
   - `raw_category` if not `graded`;
   - else `no_recommendation` if the cleaned strength is `no recommendation`, or the text matches `_NO_REC_PATTERNS`;
   - else `best_practice` if the strength or text matches `best practice|good practice statement`;
   - else `graded`.
3. **Non-graded category:** strength, direction and certainty are `None`; both statuses `UNGRADED`. Stop.
4. **Strength:** tokens `strong` → STRONG; `conditional|weak` → WEAK. Truncation repair is allowed (§7.3). No token → `strength=None`, `UNMAPPED` (e.g. ICU class `recommendation`).
5. **Direction:**
   - from the label if it contains `for`/`against`;
   - else from the text: `_AGAINST_PATTERNS` → AGAINST;
   - else FOR.
   - Direction is set only when the strength is MAPPED.
6. **Certainty:**
   - strip `certainty`, `of`, `evidence`, `quality`, `confidence in estimates of effect`;
   - map `high|moderate|low|very low`;
   - a composite `a/b` → the **lower** (existing convention, `grading.py` "moderate/low → Low");
   - `insufficient` → `None`, `UNGRADED` (OV3);
   - empty → `UNMAPPED`;
   - unrecognized → `UNMAPPED`.

```python
_AGAINST_PATTERNS = (r"\b(recommends?|suggests?)\s+against\b",
                     r"\b(recommends?|suggests?)\s+not\b",
                     r"\bshould\s+not\b", r"\bdo\s+not\s+(use|give|administer|perform)\b")
_NO_REC_PATTERNS  = (r"\bcannot\s+recommend\b", r"\bno\s+recommendation\b",
                     r"\bevidence\s+(was|is)\s+(inconclusive|insufficient)\b")
```
"rather than" and "over" are **not** against (they compare two options in favour of one).

### 7.3 Truncation repair
The GT has single leading characters dropped (`onditional`, `trong`, `igh-certainty`).
Rule: a cleaned token of length ≥ 4 that is a proper suffix of **exactly one**
vocabulary word maps to that word; otherwise it's unrecognized. This is generic
and needs no alias list.

### 7.4 Tests (`test_harmonization.py`, table-driven, `pytest.mark.parametrize`)
- **Every GT row:** run all 47 ACP + ICU GT rows (30 + 17, via `GtMode.WITH_UNGRADED`, using `raw_class`/`raw_LOE`) through `harmonize`; expectations are hand-written in `tests/fixtures/harmonization_gt_expected.csv`. Must include:
  - ACP "recommends against adding a DPP-4…" → STRONG / AGAINST / HIGH;
  - ACP "suggest against clinicians treating…" → WEAK / AGAINST / LOW;
  - ICU "We suggest against the routine use of ultrafiltration…" (class `recommendation`) → strength UNMAPPED, certainty LOW;
  - ICU "cannot recommend…" → NO_RECOMMENDATION, both UNGRADED;
  - ACP "Evidence was inconclusive…" → NO_RECOMMENDATION;
  - "moderate/low-certainty evidence" → LOW;
  - truncation forms → repaired.
- **Synthetic edge cases:**
  - empty strings everywhere;
  - unknown tokens;
  - "insufficient";
  - label direction overriding text;
  - "rather than" → FOR;
  - ambiguous truncation (suffix of 2 words) → UNMAPPED;
  - non-GRADE family → raises.
- **Invariant (property-style loop over the cases):** `status == MAPPED` ⇔ value is not None.

## 8. T6 Corpus (`evident/corpus.py`)

### 8.1 Manifest `corpus/manifest.csv`
| Column | Required | Rule |
|--------|----------|------|
| `doi` | yes | normalized (§5.3); matches `^10\.\d{4,9}/\S+$`; unique |
| `society` | yes | in `_KNOWN_SOCIETIES` (M0: ESICM, SCCM, ACP) |
| `year` | yes | integer, 1990 ≤ year ≤ current year |
| `title` | yes | non-empty |
| `topic_id` | yes | slug `^[a-z0-9-]+$` |
| `supersedes_doi` | no | if set, a DOI in the manifest with the same `topic_id` and an earlier year; no cycles |
| `grading_family` | yes | `GradingFamily` value |
| `pdf_filename` | no | default `_doi_to_filename(doi)` (same rule as `pdf_loader.py:159`) |

### 8.2 API
```python
@dataclass(frozen=True)
class ManifestEntry: ...                    # the columns above, typed

def load_manifest(path: str) -> list[ManifestEntry]       # raises ManifestError listing ALL problems with line numbers
def current_editions(entries: list[ManifestEntry]) -> set[str]   # DOIs: latest year per topic_id, minus anything superseded
def sync_to_store(entries: list[ManifestEntry], store: Store, pdf_dir: str) -> SyncReport
```
- `current_editions`: two entries with the same `topic_id` and the same max year, where neither supersedes the other → `ManifestError` (ambiguous current edition).
- `sync_to_store`: upserts every entry; computes `pdf_sha256` when the PDF file exists. It returns a `SyncReport(added, updated, missing_pdf: list[doi])` and never raises for a missing PDF; that's reported.
- Header-only manifest → `[]`. Missing header column → `ManifestError`.

### 8.3 Tests (`test_corpus.py`)
- Valid manifest round trip.
- Each validation rule fails with its line number; several errors are reported together.
- Supersession: chain A→B→C gives current = {C}; cycle → error; cross-topic supersede → error; same-year tie → error.
- `sync_to_store`: added, then updated on rerun; a missing PDF appears in the report; a duplicate PDF raises through the store's `DuplicatePdfError`.

## 9. Exit criteria and verification

| # | Criterion | Command |
|---|-----------|---------|
| 1 | All new and existing tests pass | `python -m pytest tests -m "not slow"` |
| 2 | Legacy golden replay unchanged (3 original columns, exact) | `python -m pytest tests/test_legacy_golden.py` |
| 3 | V0 prompt byte-identical | included in `tests/test_prompts.py` |
| 4 | Harmonization covers every ACP+ICU GT row, 0 unexpected UNMAPPED | `python -m pytest tests/evident/test_harmonization.py` |
| 5 | ICU loader: 3 guidelines / 17 rows (GRADED_ONLY) | `tests/test_datasets.py` |
| 6 | Store survives a mid-write failure with no partial run | `tests/evident/test_store.py` |

No GPU run is required for M0. The first real-model check is M2/T10.

## 10. Implementation order

```
Lane C (extraction/):  6.0 golden ─▶ 6.4 ICU bug (test-first) ─▶ 6.1 ─▶ 6.2 ─▶ 6.3 ─▶ 6.4 few-shot/GtMode
Lane A (evident/):     domain ─▶ store (§5) ─▶ corpus (§8)
Lane B (evident/):     domain ─▶ harmonization (§7)  [needs GtMode from Lane C for the GT-table test]
```
Lanes A and B share `domain.py`: land it first, alone. Lane B's GT-table test waits
for §6.4 `GtMode`; its synthetic tests don't.

## 11. Decisions taken in this spec (flag if you disagree)
1. **Harmonized values are computed on read, not stored** (§5.1). A mapping change needs no migration.
2. **Ungraded output format:** sentinel tokens in the grade column (`BEST_PRACTICE`, `NO_RECOMMENDATION`, certainty `NA`), not a 4th pipe column. The parser stays 3-column-compatible, and SC voting works unchanged.
3. **Composite certainty takes the lower value** (existing convention).
4. **ACP "Evidence was inconclusive…" is `no_recommendation`.** The data resolves open question 3 from the design doc; ACP has no best-practice rows in the labelled set.
5. **Ground truth has no direction labels:** "recommends against" is stored as `Strong For` after normalization. Direction comes from text for GT and extraction alike, and GT is harmonized from `raw_class`/`raw_LOE`, never from normalized values.
6. **The golden replay replaces the "±0.01 F1 legacy benchmark" exit check** in `docs/roadmap.md`. The legacy benchmark isn't reproducible (unseeded few-shot).

## 12. Risks surfaced while writing
- **Runtime:** the baseline took 2,839 s for one ACP guideline (`artifacts/ab_tests/baseline.csv`). About 100 guidelines single-pass ≈ 79 h, which exceeds the 3-night snapshot budget before SC (×3). This is input to M2/T10: thinking off, table-first, chunking.
- **ACP labelled set has 29 graded rows and 1 ungraded:** ungraded-category accuracy is effectively unmeasurable in MVP. Report it as such.
- **Sci-Hub download code in `pdf_loader.py`:** T11 must not call it.
