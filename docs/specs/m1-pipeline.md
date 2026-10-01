# Spec: M1 Pipeline End to End

Status: DONE · 2026-10-01 · Branch: `agent_explor`
Parent docs: `docs/designs/living-evidence-map.md` (decisions), `docs/roadmap.md` (M1), `docs/specs/m0-foundations.md` (store, harmonization, corpus)
Tasks: T3b legacy hooks · T4 ExtractorConfig · T7 pipeline + CLI · T11 PDF fetcher · T8a headline CSV · first real run

## 1. Goal

Connect the M0 layers into commands: `snapshot`, `validate`, `publish`, `add`,
`status`. A snapshot extracts every manifest guideline with one frozen
`ExtractorVersion`. `validate` scores it against the 12 labelled guidelines and
gates `publish` against the previous published version. `add` extracts one new
guideline with the published version. Everything runs without a GPU behind
injected fakes. M1 ends with the **minimum testable product** (§13): a real
v0 snapshot of the 12 labelled guidelines, validated, published, with a headline
CSV.

### Non-goals (M1)
- Table-first strategy (T5), FT re-evaluation (T9), choosing v1 (T10): M2.
- Figures, trends, captions, dashboard (rest of T8): M3. M1 `publish` writes one headline CSV only (§12).
- Schemes other than GRADE (M6). DOI input to `add` (design: NOT in scope).
- Running `scripts/fetch_pdfs.py` for real: M3.

## 2. Constraints

| Constraint | Consequence |
|------------|-------------|
| Python 3.9, no new dependencies | stdlib + numpy/pandas/scipy already installed; `argparse` for the CLI; `urllib` for T11 |
| User rules (global CLAUDE.md) | enums not boolean params (CLI flags map to enums at the edge); private by default; early return; constants; short what/why comments; minimal diff in `extraction/` |
| Layering | `cli → pipeline → {corpus, extraction, validation, harmonization, store}`. Only `evident/extraction.py` and `evident/validation.py` import legacy `extraction/` / `evaluation/` |
| Sci-Hub | `load_pdf_pages` falls back to a DOI download (incl. Sci-Hub, `pdf_loader.py:217`) when the path doesn't exist. `evident.extraction.run` refuses a missing file **before** calling legacy code. T11 must not import `pdf_loader` |
| No GPU in tests | every model, Ollama call and GT load sits behind `PipelineDeps` (§8.1); tests inject fakes |
| Bug fixes | test first, see it fail, then fix |
| Legacy regression gate | `tests/test_legacy_golden.py` stays green after every `extraction/` change |

## 3. Layout

```
evident/
  extraction.py   ExtractorConfig, ExtractionOutput, run(): adapter over legacy extraction/
  validation.py   labelled GT join, scoring, paired-bootstrap gate (pure apart from GT loading)
  pipeline.py     add, snapshot, validate, publish, status + PipelineDeps
  analytics.py    headline aggregates over harmonized rows (pure); T8 grows it in M3
  cli.py          argparse front end; maps flags to enums; prints reports
  __main__.py     python -m evident
scripts/
  fetch_pdfs.py   T11
tests/evident/
  test_extraction.py  test_validation.py  test_analytics.py  test_pipeline.py  test_cli.py  test_e2e.py
configs/
  v0.json         legacy baseline ExtractorConfig (§13)
corpus/
  manifest.csv    12 labelled guidelines (§13); M3 adds the rest
tests/scripts/
  test_fetch_pdfs.py
```

Dependency graph (arrows = "imports"):
```
 cli ──▶ pipeline ──┬──▶ corpus ──▶ store ──▶ domain
                    ├──▶ store
                    ├──▶ analytics ──▶ harmonization (pure)
                    ├──▶ harmonization
                    ├──▶ validation ──▶ harmonization; evaluation/; extraction.datasets (GT only)
                    └──▶ extraction ──▶ legacy extraction/ (extract_guideline, SC, few-shot, llm_client)
```
`validation` receives stored recommendations as arguments; it never reads the store.

## 4. T3b Legacy hooks (`extraction/`)

Additive and off by default, like M0 §6. Golden replay must stay green.

### 4.1 Per-call records
The store's `llm_call` rows need `prompt_sha256`, `chunk_id`, `sample_idx`, `seed`.
Today `ExtractionResult.per_page_responses` is a bare list of `LLMResponse`.
- `extractor.py`: new dataclass `CallRecord(sample_idx: int, chunk_id: int, seed: int | None, prompt_sha256: str, response: LLMResponse)`.
- `ExtractionResult` gains `call_records: list[CallRecord] = field(default_factory=list)`. `per_page_responses` is unchanged.
- `extract_guideline` and `self_consistency_extract` append one record per LLM call (`sample_idx=0` for single pass). `prompt_sha256` = sha256 of the UTF-8 prompt.
- Tests: one record per call; SC gives `n_samples × n_chunks` records with the right `(sample_idx, chunk_id, seed)`; prompt hash equals `sha256(prompt)` captured by the mock.

### 4.2 Thinking switch
`has_thinking` only tells the parser to strip `<think>`. Nothing turns qwen3 thinking off, which the snapshot budget requires (design OQ5).
- `llm_client.py`: `class ThinkMode(str, Enum): DEFAULT = "default"; ON = "on"; OFF = "off"`.
- `generate`, `_generate_chat`, `generate_json` gain `think: ThinkMode = ThinkMode.DEFAULT`. `DEFAULT` → no `think` kwarg to Ollama (byte-identical call). `ON`/`OFF` → `think=True/False` (ollama-python 0.6.1 supports it).
- `extract_guideline` and `self_consistency_extract` gain `think: ThinkMode = ThinkMode.DEFAULT`, forwarded **only when not DEFAULT** (same pattern as `seed`, M0 §6.3).
- Tests: kwargs with and without `think` (patched `ollama.Client`); the extractor forwards only when set.

### 4.3 Model digest
- `OllamaClient.model_digest() -> str`: `self._client.list()`, return the `digest` of the entry whose `model` equals `self.model`; raise `ModelNotFoundError(model)` (new, in `llm_client.py`) if absent.
- Test with a fake `list()` response.

## 5. T4 ExtractorConfig (`evident/extraction.py`)

### 5.1 Config
Frozen dataclass; every choice is an enum or a number.

```python
class ExtractionMode(str, Enum):  SINGLE_PASS = "single_pass"; SELF_CONSISTENCY = "self_consistency"   # M2 adds TABLE_FIRST
class FewShotMode(str, Enum):     ZERO_SHOT = "zero_shot"; FEW_SHOT = "few_shot"
class ConsensusMode(str, Enum):   FIXED = "fixed"; ADAPTIVE = "adaptive"
class MlFilterMode(str, Enum):    OFF = "off"; ON = "on"

@dataclass(frozen=True)
class ExtractorConfig:
    mode: ExtractionMode
    model: str
    thinking: ThinkingMode               # evident.domain; mapped to llm_client.ThinkMode
    prompt_version: str                  # PromptVersion value ("v0" | "v1")
    few_shot: FewShotMode
    n_examples: int = 3
    few_shot_seed: int = 0
    seed: int = 0                        # base LLM seed; SC sample k uses seed + k
    pages_per_chunk: int = 1
    dedup_threshold: float = 0.9
    sc_samples: int = 3
    sc_temperature: float = 0.3
    sc_consensus: ConsensusMode = ConsensusMode.ADAPTIVE
    sc_consensus_threshold: int = 2
    sc_similarity_threshold: float = 0.85
    ml_filter: MlFilterMode = MlFilterMode.OFF
    ml_filter_threshold: float = 0.3
    classifier_path: str = "artifacts/classifier/rec_classifier.joblib"
    embedding_model: str = "FremyCompany/BioLORD-2023"

    def to_json(self) -> str             # canonical: sorted keys, enum values, no whitespace
    @classmethod
    def from_json(cls, s: str) -> "ExtractorConfig"
```
Not configurable, by design: `normalize` (raw output is stored, OV3), JSON output,
verify, post-filter, grading/context oracle (closed approaches in memory notes).
SC fields are ignored in SINGLE_PASS but still hashed (one config = one version).

### 5.2 Version identity
`ExtractorVersion.id = sha256(canonical JSON of {config, model_digest, classifier_sha256, embedding_revision, few_shot_pool_sha256})`.
- `model_digest`: `OllamaClient.model_digest()` (§4.3).
- `classifier_sha256`: file hash of `classifier_path` when `ml_filter == ON`, else `None`.
- `embedding_revision`: commit hash of the BioLORD snapshot in the Hugging Face cache (`huggingface_hub`, already installed with sentence-transformers; `local_files_only`). Today `167aab52…`.
- `few_shot_pool_sha256`: hash of the canonical GT example pool for the scheme (changes if GT changes); `None` for zero-shot.
- **Not** in the id: git SHA. It is stored per run (§6) so a code commit doesn't force a re-snapshot.

```python
@dataclass(frozen=True)
class Artifacts:              # what the live machine has right now
    model_digest: str
    classifier_sha256: Optional[str]
    embedding_revision: str
    few_shot_pool_sha256: Optional[str]

def probe_artifacts(cfg: ExtractorConfig) -> Artifacts
def version_for(cfg: ExtractorConfig, artifacts: Artifacts) -> ExtractorVersion
def check_artifacts(version: ExtractorVersion, live: Artifacts) -> None   # raises ArtifactMismatchError naming the field
```
`ArtifactMismatchError` covers the design's `ModelDigestMismatch` plus classifier and embedding drift.

### 5.3 Run
```python
@dataclass(frozen=True)
class ExtractionOutput:
    recs: list[RawRecommendation]     # ordinal = row order; raw class/LOE/category/page/chunk_id
    calls: list[LlmCall]              # from CallRecord
    n_pages: int
    few_shot: list[dict]              # examples actually used (OV2)

def run(cfg: ExtractorConfig, pdf_path: str, target_doi: str, models: Models) -> ExtractionOutput
```
- Missing file → `PdfMissingError` before any legacy call (Sci-Hub guard, §2).
- Few-shot: `get_few_shot_examples(GRADE, cfg.n_examples, exclude_doi=target_doi, seed=cfg.few_shot_seed)`.
- Dispatch on `mode` to `extract_guideline` / `self_consistency_extract` with `normalize=False`, `seed`, `think`, and `models.similarity` injected as `dedup_model`/`similarity_model`/`ml_similarity_model` (4A: loaded once per process).
- `Models(similarity)` is built once by the CLI (`load_models(cfg)`); tests pass a fake encoder.

### 5.4 Tests (`test_extraction.py`)
- JSON round trip; canonical form is key-order independent; enums serialize as values.
- Version id changes with each config field and each artifact; unchanged by dict order.
- `check_artifacts` raises per field.
- `run` with a missing PDF raises and never calls `load_pdf_pages` (spy).
- `run` (golden ACP fixture, mock client): recs carry raw class/LOE (not normalized), category, page, chunk_id; calls match `call_records`; few-shot excludes the target DOI and is stored.
- SC dispatch passes `seed`, `think`, consensus mode.

## 6. Store changes (migration 2)

```sql
ALTER TABLE snapshot_member ADD COLUMN origin TEXT NOT NULL DEFAULT 'initial';   -- MemberOrigin
ALTER TABLE extraction_run  ADD COLUMN code_sha TEXT;                             -- git HEAD at run time
CREATE TABLE validation (
  id                   INTEGER PRIMARY KEY,
  snapshot_id          INTEGER NOT NULL REFERENCES snapshot(id),
  baseline_snapshot_id INTEGER REFERENCES snapshot(id),
  gate                 TEXT NOT NULL,          -- GateResult
  report_json          TEXT NOT NULL,          -- full ValidationReport
  created_at           TEXT NOT NULL
);
```
Domain additions: `MemberOrigin(str, Enum)`: `INITIAL`, `RESCAN`, `POST_PUBLISH`; `GateResult(str, Enum)`: `PASS`, `FAIL`, `NO_BASELINE`; frozen `SnapshotMember(guideline_id, excluded_reason, origin)`; `Validation(id, snapshot_id, baseline_snapshot_id, gate, report_json, created_at)`.

New / changed `Store` API:
```python
def add_snapshot_member(self, snapshot_id, guideline_id, origin: MemberOrigin = MemberOrigin.RESCAN)  # BUILDING only (unchanged rule)
def add_published_member(self, snapshot_id, guideline_id) -> None        # PUBLISHED only; origin POST_PUBLISH (used by `add`)
def transition_snapshot(self, snapshot_id, to, accept_reason: Optional[str] = None)  # reason stored on → PUBLISHED
def start_run(self, snapshot_id, guideline_id, thinking, few_shot, code_sha: Optional[str] = None) -> int
def get_snapshot(self, snapshot_id) -> Optional[Snapshot]
def members(self, snapshot_id) -> list[SnapshotMember]
def runs(self, snapshot_id) -> list[ExtractionRun]                        # all statuses, for status/ETA
def save_validation(self, snapshot_id, baseline_snapshot_id, gate, report_json) -> int
def latest_validation(self, snapshot_id) -> Optional[Validation]
```
`create_snapshot` writes `INITIAL`. Tests: migration 1→2 on a v1 file keeps rows; `add_published_member` refused outside PUBLISHED; origin round trip; accept reason stored; validation round trip.

## 7. Validation (`evident/validation.py`)

### 7.1 Labelled set and DOI join
```python
@dataclass(frozen=True)
class LabelledGuideline:
    doi: str                          # normalize_doi
    family: GradingFamily
    gt: list[GtRow]                   # GtRow(text, raw_class, raw_loe, category)

def load_labelled() -> list[LabelledGuideline]   # ACP + ICU via GtMode.WITH_UNGRADED; must be 12
```
`score(...)` receives `{doi: list[RawRecommendation]}` (active runs of the snapshot).
Every labelled DOI must be present → else `ValidationJoinError(missing_dois)` (OV4 hard fail).

### 7.2 Scoring
Per guideline, both sides are harmonized on read (`harmonize`, raw columns only):
- **Graded-only completeness:** extracted rows with harmonized category `graded` vs GT rows with category `graded`; `evaluation.match_recommendations(similarity_threshold=0.65)` (same as `benchmark.run_full_benchmark`); counts TP/FP/FN.
- **Grade accuracy** on matched graded pairs: `strength_ok` = equal strength and direction, both MAPPED; `certainty_ok` = equal certainty, both MAPPED; `combined_ok` = both.
- **Ungraded P/R:** the same matching on rows with category ≠ graded. Reported, never gated (OV4).
- Pooling is micro (sum counts over guidelines) — per-guideline F1 is unstable at 1–6 recs.

```python
@dataclass(frozen=True)
class GuidelineScore: doi; tp; fp; fn; n_matched; n_strength_ok; n_certainty_ok; n_combined_ok; ungraded_tp; ungraded_fp; ungraded_fn
@dataclass(frozen=True)
class SnapshotScore: per_guideline: tuple[GuidelineScore, ...]   # sorted by doi
def score(recs_by_doi, labelled, similarity_model) -> SnapshotScore
```

### 7.3 Gate (OV4)
```python
def gate(candidate: SnapshotScore, baseline: Optional[SnapshotScore], n_boot=2000, seed=0) -> GateReport
```
- No baseline → `NO_BASELINE` (first publish, i.e. v0).
- Paired guideline-cluster bootstrap: resample guideline indices with replacement, same draw for both versions; per draw compute micro F1 and micro combined accuracy for each; diff = candidate − baseline.
- Fails iff the 97.5th percentile of the diff is below `-0.02` for **F1 or combined accuracy**.
- `GateReport` carries point estimates, per-version 95% CIs (unpaired bootstrap) and diff CIs, strength/certainty accuracy, ungraded P/R; serialized into `validation.report_json`.
- Constants: `_GATE_MARGIN = 0.02`, `_N_BOOTSTRAP = 2000`, `_CI_PERCENTILES = (2.5, 97.5)`, `_MATCH_THRESHOLD = 0.65`.

### 7.4 Tests (`test_validation.py`)
- Join: one labelled DOI missing → `ValidationJoinError` naming it.
- Scoring on hand-built recs with the fake similarity model: TP/FP/FN, strength vs certainty vs combined, ungraded counted separately, harmonization from raw (an "against" text with label "Strong" scores direction AGAINST on both sides).
- Gate: identical scores → PASS; candidate clearly worse on every guideline → FAIL; worse on F1 only → FAIL; small noise → PASS; no baseline → NO_BASELINE; same seed → same report.
- `load_labelled()` returns 12 guidelines / 46 rows (real data, not slow).

## 8. T7 Pipeline (`evident/pipeline.py`)

### 8.1 Dependencies
```python
@dataclass(frozen=True)
class PipelineDeps:
    store: Store
    manifest_path: str
    pdf_dir: str
    runner: Callable[[ExtractorConfig, str, str], ExtractionOutput]   # default: extraction.run with loaded Models
    probe: Callable[[ExtractorConfig], Artifacts]                     # default: extraction.probe_artifacts
    labelled: Callable[[], list[LabelledGuideline]]                   # default: validation.load_labelled
    similarity_model: object                                          # for validation matching
    code_sha: Callable[[], Optional[str]]                             # git rev-parse HEAD, None outside git
```
The CLI builds the real one; tests build fakes. This is the "strategy interface is a function signature" decision (eng review).

### 8.2 snapshot
```python
class SnapshotStart(str, Enum): NEW = "new"; RESUME = "resume"
def snapshot(deps, start: SnapshotStart, config: Optional[ExtractorConfig] = None,
             excludes: Sequence[tuple[str, str]] = ()) -> SnapshotReport
```
1. **NEW:** refuse if a BUILDING snapshot exists (`SnapshotInProgressError`, hint `--resume`). `load_manifest` + `sync_to_store`; probe artifacts; register version; `create_snapshot` with **all** manifest guidelines (trends use all editions).
2. **RESUME:** latest BUILDING snapshot; else `NoSnapshotInProgressError`. Rebuild config from the stored version, probe, `check_artifacts` (refuse on drift).
3. Apply `excludes` (`(doi, reason)`) via `exclude_member`.
4. For each pending member, in guideline id order: `start_run` → `runner` → `finish_run`. Any exception except `KeyboardInterrupt`/`SystemExit` → `fail_run(error)` and continue. A missing PDF fails the run (`PdfMissingError`); it is **never** auto-excluded.
5. **Re-scan (OV5):** reload the manifest, sync, add new guidelines as `RESCAN` members, extract them (step 4).
6. If `pending_members() == []` → `COMPLETE`. Otherwise stay BUILDING and report what blocks it.

`SnapshotReport(snapshot_id, state, n_succeeded, n_failed, n_failed_empty, pending: list[doi], failures: list[(doi, error)])`.

### 8.3 validate
```python
def validate(deps, snapshot_id: Optional[int] = None) -> ValidationReport
```
Target: given id, else latest COMPLETE. Baseline: latest PUBLISHED (its active runs). Reads active-run recs per DOI, scores both, gates, `save_validation`. Re-validation appends a new row; publish reads the latest.

### 8.4 publish
```python
def publish(deps, accept_regression: Optional[str] = None) -> PublishReport
```
- A COMPLETE snapshot exists:
  - needs a validation row → else `ValidationMissingError`;
  - coverage (OV5): every manifest guideline is a member, included members all have an active run → else `CoverageError(dois)`;
  - gate `FAIL` and no reason → `GateFailedError(report)`; with a reason → publish and store it;
  - `transition_snapshot(PUBLISHED, accept_reason)`.
- Both paths end by writing the headline CSV for the published snapshot (§12). Re-publishing the latest PUBLISHED snapshot (after `add`) only rewrites outputs.
- Neither → `NothingToPublishError`.

### 8.5 add
```python
class IngestMode(str, Enum): NORMAL = "normal"; FORCE = "force"
def add(deps, pdf_path: str, mode: IngestMode = IngestMode.NORMAL) -> AddReport
```
1. Latest PUBLISHED snapshot → else `NoPublishedSnapshotError` (add never changes version).
2. Find the manifest entry whose `pdf_filename` equals the PDF's basename → else `ManifestEntryMissingError` ("add the manifest row first"). Copy the PDF into `pdf_dir` if it lives elsewhere.
3. Sync that entry (`DuplicatePdfError` propagates if the bytes belong to another DOI).
4. Already an active run in the snapshot and `NORMAL` → no-op, `AddReport(status=ALREADY_PRESENT)`. `FORCE` → new run; the later SUCCEEDED run becomes active.
5. `add_published_member` if not a member; probe + `check_artifacts`; `start_run` → `runner` → `finish_run` / `fail_run`.
The CLI chains `publish` after a successful `add` (design: "add = ingest then publish").

### 8.6 status
```python
def status(deps) -> StatusReport
```
For the latest BUILDING snapshot if any, else the latest PUBLISHED:
- counts: members, excluded (with reasons), succeeded, failed, failed-empty, pending, **interrupted** (RUNNING runs older than the newest run of the same member, or any RUNNING run when no process holds the snapshot; M1 treats every RUNNING run as interrupted since runs are sequential);
- **ETA** = median `duration_s` of SUCCEEDED runs × pending count (`None` with no succeeded run);
- **slow:** SUCCEEDED runs with `duration_s > 3 × median` (4B);
- **low-recall outliers:** active-run rec count `< 0.5 ×` or ungraded share `> median + 0.25` versus the society median (constants);
- **coverage (OV5):** manifest guidelines that are not members or lack an active run;
- latest validation gate for the snapshot, if any.

### 8.7 Errors
All pipeline errors subclass `PipelineError` (in `pipeline.py`), each a named class. `StoreError`, `ManifestError`, `ArtifactMismatchError`, `ValidationJoinError` propagate unchanged; the CLI prints them.

### 8.8 Tests (`test_pipeline.py`, tmp DB + tmp manifest + fake deps)
- snapshot NEW: all members extracted → COMPLETE; a raising runner → FAILED run, snapshot stays BUILDING, other members still run.
- RESUME after a simulated crash (a RUNNING run left behind) → only pending members re-run; status counts the interrupted run.
- `excludes` unblock COMPLETE; a missing PDF fails, is not excluded.
- re-scan picks up a manifest row added during the build (runner adds it on first call).
- artifact drift on RESUME → `ArtifactMismatchError`, no runs started.
- validate: join failure; NO_BASELINE first; second snapshot compared to published.
- publish: missing validation, FAIL refused, FAIL + reason stored, coverage error, re-publish path.
- add: no published snapshot; manifest row missing; ALREADY_PRESENT no-op; FORCE makes the new run active; member origin POST_PUBLISH; duplicate PDF bytes raise.
- status: ETA, slow flag, outlier flag, coverage gap.

## 9. CLI (`evident/cli.py`, `python -m evident`)

```
evident [--db PATH] [--manifest PATH] [--pdf-dir PATH] <command>
  snapshot --config CONFIG.json [--exclude DOI --reason TEXT]...
  snapshot --resume             [--exclude DOI --reason TEXT]...
  validate [--snapshot ID]
  publish  [--accept-regression REASON]
  add PDF  [--force]                       # then publish
  status
```
- Defaults are module constants: DB `out/evident.sqlite`, manifest `corpus/manifest.csv`, PDF dir `/mnt/data1/klug/datasets/evidence_extraction/pdfs`.
- Flags map to enums at the edge (`--force` → `IngestMode.FORCE`, `--resume` → `SnapshotStart.RESUME`).
- `--exclude`/`--reason` must pair up; unpaired → usage error.
- Models (BioLORD, classifier) load lazily, once per process, only for commands that extract or validate (4A).
- Exit codes: `0` ok, `1` known error (`PipelineError`, `StoreError`, `ManifestError`, …, message only, no traceback), `2` usage error (argparse).
- Tests (`test_cli.py`): parsing → the right pipeline call with the right enums (pipeline patched); error → exit 1 and message; `add` chains `publish`.

## 10. T11 PDF fetcher (`scripts/fetch_pdfs.py`)

```
python scripts/fetch_pdfs.py --email YOU@ORG [--manifest corpus/manifest.csv] [--pdf-dir DIR] [--report fetch_report.csv]
```
Per manifest entry, in order:
1. file already in `pdf_dir` and valid → `PRESENT`;
2. Unpaywall `best_oa_location.url_for_pdf` (the `--email` is required by Unpaywall; passed per run, never stored or defaulted);
3. landing page via `https://doi.org/<doi>` → `<meta name="citation_pdf_url">` (works from inside an institutional VPN);
4. otherwise `PAYWALLED` (HTTP 401/403 or a non-PDF body on the PDF URL) or `NOT_FOUND`.

- Validity: body starts with `%PDF`; written to `<name>.part` then renamed (no half files); SHA-256 recorded.
- `FetchStatus(str, Enum)`: `PRESENT`, `DOWNLOADED`, `PAYWALLED`, `NOT_FOUND`, `INVALID_PDF`.
- Report CSV: `doi, status, source, sha256, detail`. Idempotent: a rerun only fetches what is missing.
- Uses `evident.corpus.load_manifest` and `utils.doi.doi_to_filename`. Never imports `extraction.pdf_loader`. 1 s delay between requests (constant).
- Tests (`tests/scripts/test_fetch_pdfs.py`): HTTP mocked at a single private `_http_get(url) -> (status, content_type, body)`; one test per status; a `.part` file never survives a failure; rerun skips PRESENT; the email appears only in the Unpaywall query.

## 11. Exit test (`tests/evident/test_e2e.py`, no GPU)

A tmp manifest with 3 guidelines (2 labelled by a fake `labelled()`), tmp PDFs (small byte files), a fake runner returning canned recs, and the fake similarity model:
```
snapshot NEW (runner fails guideline 3)  → BUILDING, status: 1 failed, ETA set
snapshot RESUME                          → COMPLETE, status shows the earlier failure as history
validate                                 → NO_BASELINE
publish                                  → PUBLISHED, out/headline.csv written
manifest + row 4; add pdf4               → POST_PUBLISH member, active run; publish re-publishes
snapshot NEW with a worse fake config    → COMPLETE
validate                                 → FAIL
publish                                  → GateFailedError
publish --accept-regression "test"       → PUBLISHED, reason stored
status                                   → clean: no pending, no coverage gap
```
The roadmap's order (`add → snapshot → validate → publish`) can't run as written: `add` needs a published version. This order exercises the same commands.

## 12. T8a Headline CSV (`evident/analytics.py`)

The first slice of T8, so real output can be inspected at the end of M1.

### 12.1 Rows
`pipeline.publish` collects, for each **current edition** (`corpus.current_editions`) that is an included member with an active run, every stored recommendation harmonized on read:
```python
@dataclass(frozen=True)
class HarmonizedRow: doi; society; year; grade: HarmonizedGrade
def headline(rows: list[HarmonizedRow]) -> list[HeadlineRow]    # pure
```

### 12.2 Table (`out/headline.csv`, one row per society plus `ALL`)
| Column | Definition (design denominator rule) |
|--------|--------------------------------------|
| `society` | society, or `ALL` |
| `n_guidelines` | guidelines contributing rows |
| `n_recs` | graded + ungraded |
| `n_graded`, `n_ungraded`, `ungraded_share` | by harmonized category |
| `pct_strong` | STRONG / graded with strength MAPPED |
| `pct_against` | AGAINST / graded with strength MAPPED |
| `pct_high_moderate` | HIGH or MODERATE / graded with certainty MAPPED |
| `n_strength_unmapped`, `n_certainty_unmapped`, `n_certainty_ungraded` | excluded from the % denominators, reported (OV8) |
| `snapshot_id`, `extractor_version_id` | provenance |

- Percentages are rounded to one decimal; an empty denominator gives an empty cell, never 0.
- Path constant `_HEADLINE_PATH = "out/headline.csv"`; written atomically (tmp + rename).
- Excluded members and non-current editions are not counted; their count goes into `PublishReport`.

### 12.3 Tests (`test_analytics.py`)
- Hand-built rows: every column, including unmapped exclusion from denominators and ungraded share.
- Empty denominators → empty cells; `ALL` equals the sum of societies.
- `publish` writes the file for the published snapshot only; superseded editions are skipped.

## 13. First real run: minimum testable product (GPU, one night)

After §11 is green, run the pipeline for real on the 12 labelled guidelines. This
is v0 for M2's comparisons and the first measurement of runtime with thinking off.

### 13.1 Inputs
- `corpus/manifest.csv`: 12 rows (9 ACP, 3 ESICM), from `ACP.csv` and the ICU xlsx metadata (DOI, title, year). `topic_id` hand-picked; the 3 ESICM fluid-therapy parts are separate topics. Each row validates; every PDF is on disk.
  `test_corpus.test_repo_manifest_is_header_only` becomes `test_repo_manifest_is_valid` (loads, 12 rows, all current editions).
- `configs/v0.json`: the legacy baseline as an `ExtractorConfig`:
  `SINGLE_PASS`, `qwen3:14b`, thinking `OFF`, prompt `v0`, `FEW_SHOT` with 3 examples, `pages_per_chunk=1`, ML filter `OFF`, `seed=0`, `few_shot_seed=0`.

### 13.2 Procedure
```
python -m evident snapshot --config configs/v0.json     # overnight
python -m evident status                                 # ETA, failures, slow flags
python -m evident snapshot --resume                      # if anything failed or it was interrupted
python -m evident validate
python -m evident publish
```

### 13.3 Recorded results
Append to `docs/progress.md` (and the memory notes):
- validate: graded F1, P, R, strength / certainty / combined accuracy with 95% CIs, ungraded P/R;
- per-guideline duration (median, max) and the projected hours for ~100 guidelines;
- whether `think=OFF` took effect (check `eval_tokens` and the absence of `<think>` in stored `raw_text`);
- the headline CSV;
- any failure and how it was resolved.

### 13.4 Done when
- The snapshot is PUBLISHED with 12/12 active runs and no exclusions (or each exclusion justified).
- `out/headline.csv` exists and its counts match `status`.
- Results are recorded (§13.3). A projected full-corpus time over ~3 nights is a finding for M2, not an M1 failure.

## 14. Exit criteria and verification

| # | Criterion | Command |
|---|-----------|---------|
| 1 | All tests pass, legacy golden unchanged | `python -m pytest tests -m "not slow"` |
| 2 | E2E add/snapshot/validate/publish/status with fakes | `python -m pytest tests/evident/test_e2e.py` |
| 3 | Resumed snapshot shown correctly by `status` | included in §11 and §8.8 |
| 4 | Missing PDF never reaches `load_pdf_pages` | `tests/evident/test_extraction.py` |
| 5 | Fetcher statuses with mocked HTTP | `python -m pytest tests/scripts` |
| 6 | Smoke (manual, no GPU): `python -m evident status` on an empty DB prints "no snapshot" and exits 0 | shell |
| 7 | Headline CSV columns and denominators | `python -m pytest tests/evident/test_analytics.py` |
| 8 | Minimum testable product: real v0 snapshot of the 12 labelled guidelines published, results recorded | §13 (GPU, manual) |

## 15. Implementation order

```
Lane C (extraction/):  4.1 call records ─▶ 4.2 think ─▶ 4.3 digest ─▶ T4 evident/extraction.py
Lane A (evident/):     store migration 2 (§6) ─▶ validation (§7)
Lane D (scripts/):     T11 fetcher (independent)
Lane B (evident/):     analytics headline (§12, pure; needs only harmonization)
Then:                  pipeline (§8, needs A + B + C) ─▶ CLI (§9) ─▶ E2E (§11)
Last:                  manifest rows + configs/v0.json ─▶ real run (§13, GPU overnight)
```
One commit per numbered section. Run the golden replay after each Lane C step.

## 16. Decisions taken in this spec (flag if you disagree)
1. **`add` attaches to the published snapshot** through a `POST_PUBLISH` member (migration 2), instead of a snapshot-less run. The snapshot stays the unit of "one version"; the paper can list late additions.
2. **Git SHA is stored per run, not in the version id.** Otherwise every commit would force a full re-snapshot.
3. **ExtractorConfig drops `normalize`, JSON output, verify, post-filter and the oracles.** Raw output is stored (OV3); the rest are closed approaches. SC and the ML filter stay: they are M2 candidates.
4. **A missing PDF fails the run**, it is not auto-excluded. Gaps stay visible until fixed or explicitly excluded with a reason.
5. **Gate metrics:** micro graded-only F1 and micro combined harmonized accuracy (strength + direction + certainty). Strength and certainty accuracy are reported separately but not gated, to limit false alarms on 12 guidelines.
6. **GT mode for validation is fixed to `WITH_UNGRADED`**, split by category, so graded and ungraded scores come from one load.
7. **Matching threshold 0.65** (BioLORD), the value the legacy benchmark used.
8. **`publish` in M1 gates, records and writes one headline CSV.** Figures, trends, captions and the dashboard arrive with the rest of T8 in M3.
9. **`ThinkMode` lives in `llm_client`** with a `DEFAULT` that omits the kwarg; `evident` maps its `ThinkingMode` onto it. `extraction/` stays independent of `evident/`.
10. **Unpaywall email is a required CLI argument**, never a default in code.
11. **The headline counts current editions only**, per the design ("current map uses the latest edition"). Trends over all editions come with T8.
12. **The real run uses prompt V0**, so v0 matches the legacy baseline. V1 (ungraded categories) is compared against it in M2; v0's ungraded share will be near zero.

## 17. Risks
- **Real-run runtime.** The legacy baseline took 2,839 s for one ACP guideline. 12 guidelines can exceed one night if thinking isn't off; `--resume` makes splitting over two nights safe.
- **`think=False` on qwen3 via Ollama 0.34** is untested on this machine. The real run (§13.3) checks it; if it is ignored, the snapshot budget (≈3 nights) fails.
- **Bootstrap on 12 clusters** gives wide CIs; the gate catches only clear regressions. That is the intent of OV4, but say so in the paper.
- **`load_pdf_pages` DOI fallback** stays in legacy code. Only the `run()` guard protects the pipeline; a future direct caller could still reach Sci-Hub.
- **BioLORD revision** comes from the local HF cache. A cache wipe plus re-download of a newer revision is caught by `check_artifacts`, but blocks `add` until a new snapshot.

## 18. Implementation notes (2026-10-01)
All in, including the real run (results in `docs/progress.md`). Differences from the text above:
- **Few-shot examples are stored at `finish_run`.** The runner picks them, so `start_run` gets `[]` and `finish_run(..., few_shot=)` overwrites it.
- **Pool drift is detected through the id.** `ExtractorVersion` has no pool column; `check_artifacts` compares digest, classifier and embedding field by field, then recomputes the id. A different id with equal fields is reported as `few_shot_pool_sha256`.
- **`PipelineDeps.headline_path`** added so tests write to a tmp dir; the CLI passes `out/headline.csv`.
- **`status` reports the newest snapshot of any state**, so a COMPLETE snapshot awaiting `publish` is visible. Only one can be BUILDING, so this is always the one in progress.
- **`add` probes artifacts before attaching the member**, so a drift doesn't leave a member without a run. A failed `add` exits 1 and doesn't publish.
- **`ExtractorConfig` rejects any embedding model but BioLORD**, so the CLI's single lazy BioLORD serves extraction and validation.
- **Validation matching** uses `build_similarity_matrix` + Hungarian directly (same as `match_recommendations`) to keep row indices for grade comparison.
- **Dedup differs from the legacy baseline.** §5.3 injects BioLORD as `dedup_model`; `benchmark.run_full_benchmark` ran exact-only dedup. v0 may drop a few near-duplicates the legacy run kept.
