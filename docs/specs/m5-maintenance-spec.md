# Spec: M5 — Maintenance mode for the living evidence map

**Status:** Code implemented 2026-10-26 (§9); first live release 2026-10-27; exit demonstration pending  
**Repository reviewed:** `JulianKlug/evident`, `main` at `a1ba3b85b8c04d168ee428e44e2b553270b7aec0`  
**Sources of truth:** `docs/roadmap.md` (M5), `docs/designs/living-evidence-map.md` (Decisions 4, 5, 8–10), `docs/specs/m4-paper1.md`, `docs/paper1/freeze.md`, `docs/progress.md`, and `TODOS.md`.

## 1. Goal

Make the published ESICM/SCCM/ACP evidence map safe to maintain after the M4 paper freeze. An operator can find and curate a guideline, add **only that guideline** with the currently published ExtractorVersion, resolve review requirements, and regenerate a trustworthy current dashboard without re-extracting the corpus. A replacement edition changes the current view but not the historical record. An extractor change requires a separately validated full snapshot.

**M5 is an operational hardening milestone, not a new extraction or analytical methods milestone.** A successful update must be observable, reproducible, and recoverable; an incomplete update must never silently appear as complete.

## 2. Starting point: reuse what exists

- **M4 source is frozen.** Snapshot **8**, v2 ExtractorVersion `39ac17093b65…`, contains **85 guidelines, 75 current and 10 superseded** in the M4 corpus. Its read-only database and October 2026 archive are described in `docs/paper1/freeze.md` (the freeze itself is dated **2026-10-13** and the package archive **2026-10-25**). Keep the `paper1-freeze` tag, frozen DB, M4 artefacts, and `configs/paper1.json` independent of subsequent live-map changes. Do not assume the live machine's DB is identical to the archived copy without checking.
- `evident add PDF` **already extracts using the published version and automatically calls `publish`** in `evident/cli.py`. The roadmap's extra `evident publish` after a successful add is redundant in the current CLI. `add --force` re-extracts an existing guideline. DOI/PDF duplicate handling and run provenance already exist.
- `evident snapshot --config/--resume`, `validate`, `publish`, `reject`, `status`, corpus manifests, edition resolution, and staged table/figure/dashboard rendering already exist. Keep their interfaces and semantics unless a correctness issue demands a small extension.
- The M4 within-guideline deduplication uses a screened pair set, two-model consensus, and a **frozen** `corpus/dedup/paper1_verdicts.csv`. Ordinary live `publish` currently **warns but keeps both** when new screened pairs have no verdict. The CLI default points to the frozen M4 verdict file; the offline adjudication script hardcodes snapshot 8. These are M5 gaps.
- Current `publish` re-publication of a PUBLISHED snapshot skips the coverage gate used for a COMPLETE candidate. `status` displays the latest snapshot by ID, which may be an unfinished candidate instead of the current public release. PDF hash drift is reported, not blocked; `add` without `--force` may return already-present for a changed file.
- `corpus/themes/assignments.csv` can leave new recommendation texts under the explicit **Not yet assigned** theme. This is acceptable only when disclosed and triaged, not silently ignored.

## 3. Scope and non-goals

**In scope:** one-guideline ingestion; corpus audit and PDF acquisition; same-version incremental updates; edition supersession; post-add deduplication; pre-publication health checks; live release provenance; runbook; upgrade/recovery procedure; and integration tests proving the M5 exit criterion.

**Out of scope:** M6 societies and new grading schemes; v3 ungraded extraction; full automation of literature screening; a new end-user UI; rewriting the M4 paper package; general recommendation-level supersession; and public deployment as a prerequisite. Public hosting remains a separate approval decision in `TODOS.md`.

Keep the **graded-only** interpretation of headline results. Do not promote the v2 ungraded share to a reliable result. Exploratory themes may display an unassigned category until reviewed.

## 4. Behavioural contract

### 4.1 Discover and curate a guideline

1. Refresh `corpus/candidates.csv` with `scripts/search_guidelines.py --until YYYY-MM-DD`, preserving manual decisions. Check the society's guideline pages, co-publications, corrections, first-online date, GRADE eligibility, and whether the paper is a full guideline or a derivative.
2. For an included item, add one `corpus/manifest.csv` row with normalized DOI, `society`, `year`, `title`, `topic_id`, `grading_family`, `pdf_filename`, optional `joint_with`, and optional `supersedes_doi`. Set its candidate `decision=include`. Re-run `load_manifest` and `check_candidates` checks **before any extraction or publication**; the live path should also enforce parity, not rely solely on tests.
3. Fetch legally accessible PDFs using `scripts/fetch_pdfs.py`, optionally within the institutional VPN. If blocked, use `scripts/import_pdfs.py missing` and `import` after an authorized manual download. Log `blocked`, `paywalled`, and `not_found` outcomes; do not mark an unresolved item published. Verify the file is a readable, DOI-correct guideline PDF and has the manifest filename.
4. Record the inclusion decision, any exclusion or delay reason, the fetch source, and the search cutoff. `search_guidelines.py` writes `corpus/search.json` (`{"start": "YYYY-MM-DD", "cutoff": "YYYY-MM-DD"}`, tracked in git) next to `candidates.csv`; live `publish` reads the cutoff from it (§5). Do not invent a `supersedes_doi` link from a similar title alone.

### 4.2 Routine add: same version, one run

1. Preflight the manifest/candidates, file existence/identity, current published snapshot, local model digest, and embedding revision. Reject conflicting DOI/PDF hashes. An unchanged DOI with an active matching-PDF run is idempotent; a changed PDF **requires explicit `--force` or `rebind-pdf` (§4.2.6)** and must not be called already present. `fetch_pdfs.py` and `import_pdfs.py` already leave an existing valid PDF untouched, so drift comes only from a manual replacement or a lost file (see §5 gate 3).
2. `evident add PDF` creates at most one new successful active run for the target DOI using the **published** snapshot's ExtractorVersion. It does not re-run existing guideline PDFs. Failed or empty extraction is visible in `status`; the last successfully rendered public outputs remain intact.
3. Screen the newly active recommendations for within-guideline duplicates using the existing M4 algorithm. Existing reviewed pairs are reused **only if their DOI, ordinals, and both statement texts still match** (exact string equality on `text_a`/`text_b`, no normalization; a whitespace change means a new judgment). Every other screened pair requires two local judges using the established consensus rule: merge only if both vote duplicate and harmonized mapped grades do not conflict. Save verdicts with model digests and audit text. Do not infer distinct merely because a verdict is missing.
4. **Proposed small CLI change:** retain `add` auto-publishing when all gates pass; if new duplicate pairs need review, finish ingestion but **do not replace public outputs**. Print the affected DOI/pairs and the follow-up command sequence, and exit with a dedicated code **3 (`needs review`)**: distinct from 0 (published), 1 (error or failed gate) and 2 (argparse usage). `add --no-publish` exits 0 after a successful ingestion. Allow `add --no-publish` to stage an update explicitly. After reviewing pairs, run `evident publish --snapshot <published-id>`; no new extraction is needed.
5. Publishing recomputes tables, figures, captions and dashboard from the active successful runs of **one version**. A repeated no-op add or publish with identical inputs must yield identical generated artefact bytes (tables, figures, dashboard) and the same release identifier; it writes no new release record (§5). The dashboard clearly identifies the release, source cutoff, version, and current-versus-superseded rule.
6. **Proposed new command `evident rebind-pdf PDF`:** accept new bytes for a manifest PDF without re-extraction, when they give the model identical input. It rebuilds every prompt of the target DOI's active run from the new PDF with the run's ExtractorVersion (same chunking, seeded few-shot selection and SC sample seeds) **without calling the model**. It compares each prompt's SHA-256 with the run's stored `llm_call.prompt_sha256`, matched by chunk and sample.
   - All match, same count: update the PDF SHA-256 recorded for the guideline and the active run, keep the run, recommendations and duplicate verdicts, record the old/new hash and time in the run's provenance, exit 0.
   - Any mismatch, or a different chunk count: change nothing, list the first differing chunk, exit 1 and point to `add --force`.
   - The DOI/PDF-hash uniqueness check (`DuplicatePdfError`) still applies. Refuse a PDF without an active run, or a run whose ExtractorVersion artefacts no longer match their digests.
   - Typical outcome: metadata-only drift (XMP, object order) rebinds. A visible download stamp ("Downloaded from … on <date>") changes the text, so it needs `--force`. That is correct, since the model's input really differs.
   - Prompt construction must be callable without the LLM. If the legacy extractor interleaves the two, add a narrow prompt-only entry point; do not change prompt text.

### 4.3 Separate the live adjudications from the paper freeze

- Do **not** append to `corpus/dedup/paper1_verdicts.csv`, which is pinned by checksum in `configs/paper1.json`. Create `corpus/dedup/live_verdicts.csv`, initially seeded from the frozen decisions. Point ordinary live `publish` at it; keep `publish --package` on the M4 frozen file and frozen inputs.
- Generalise `scripts/paper1_dedup_adjudicate.py` to accept `--snapshot`, `--manifest`, `--db`, and `--dois`, and use distinct output files for each judge. Generalise or wrap `scripts/paper1_dedup_consensus.py` to merge **new pairs without discarding existing valid live verdicts**. Preserve the current conservative two-judge and grade-compatibility logic.
- Re-extraction may change ordinals. Reject a stale verdict whose pair texts do not match, even if the DOI/ordinal keys coincide; recalculate and rejudge it. Reject conflicting duplicate entries. Frozen M4 judging records remain reproducible from the `paper1-freeze` tag.

### 4.4 Updated editions and focused updates

- For a genuine whole-guideline replacement, give the new edition the same `topic_id` and set `supersedes_doi` to the previous DOI. Preserve the old guideline, PDF, and run. The current cross-section counts the newest edition once; `trends.csv` and recommendation exports retain historical editions.
- For a **focused or partial update**, do not silently supersede the entire earlier guideline. Follow the M4 convention of retaining both as current when only some questions are revised, documenting possible double counting. Recommendation-level replacement remains a separately scoped TODO.
- Reject unknown supersession references, cycles, cross-topic links, year-order violations, an ambiguous current edition, or an older edition no row supersedes (§9). In the review record show the old/current counts and any deliberate focused-update exception.

### 4.5 Extractor upgrade

- A new model, prompt, few-shot pool, extraction algorithm, grading mapping affecting results, or other ExtractorVersion dependency is **not** an `add --force` operation. Create a new version with `snapshot --config ...`, re-extract **all manifest editions**, resume failures if required, then `validate --snapshot ID` against the current published baseline.
- Honour the existing regression gate and its explicit, recorded `--accept-regression REASON` override only after human review. Do not change the published release if validation fails, coverage is incomplete, artefacts drift, or new duplicate pairs are unreviewed. Apply or rejudge live duplicate verdicts against the new snapshot's actual statement texts before publication. Re-extraction changes most texts, so expect to rejudge nearly every screened pair; the adjudication script first prints the count of pairs needing judgment (dry run) so the operator can budget GPU time before running both judges.
- Publish the completed candidate explicitly, record both version IDs, gate results, time/cost and change in the headline estimates. Keep the old published snapshot and its artefacts available for audit. The live dashboard must never combine recommendations extracted by different versions.

## 5. Publication gates, status, and provenance

**Before writing any live public outputs,** check all of the following for the target published or candidate snapshot:

1. `candidates.csv` includes and `manifest.csv` exactly agree; manifest and supersession links validate.
2. Every included, non-excluded manifest DOI is a snapshot member with one active successful run; any deliberate exclusions have recorded reasons. A published snapshot must not bypass this coverage check on re-publication.
3. The active runs match the snapshot's single ExtractorVersion, their recorded PDF SHA-256s match present files, and all required extractor artefacts match their frozen digests. Missing/changed PDFs block publication rather than merely warn. The gate itself stays strict on bytes. Re-downloaded PDFs often differ only by publisher metadata, yet the run was extracted from the recorded bytes, so harmless drift must be proven, not assumed. Resolve by (a) restoring the recorded bytes from backup, verified against the run's SHA-256; (b) `evident rebind-pdf PDF` when only the bytes changed, not the extracted input (§4.2.6); or (c) `add --force` to deliberately re-extract from the new bytes, which re-enters duplicate review.
4. Every **currently screened** duplicate pair has exactly one applicable verdict bound to its statement texts; the verdict file has no conflicting keys. Zero unresolved pairs is a release requirement. Previously judged pairs that disappear after force re-extraction may remain in the historical ledger but must not be applied to new text.
5. `corpus/search.json` exists and parses; its cutoff is not later than the publication date.
6. Generated outputs are staged first; a preflight or rendering failure preserves the previous public tables, figures and dashboard. Report a clear nonzero exit code and next action.

**Small status extension:** retain existing `status` output, add `status --published --check` (proposed) to target the latest published snapshot even while another is BUILDING, and return nonzero for any blocking gate above. Show snapshot ID/version, included/current/superseded counts, succeeded/pending/failed/empty/interrupted runs, coverage gaps, changed PDFs, unresolved duplicate-pair count, and release identity. Slow and low-recall outliers remain review warnings, not automatic hard failures; report the count assigned to **Not yet assigned** in the optional theme figure.

**Release record:** on successful live publication, write a machine-readable `out/release.json` with a `schema_version` field (start at 1). Include:

- **Release identifier:** the SHA-256 of the canonical JSON of every field below except the publication time. Identical inputs and outputs therefore give the same ID.
- **Publication time:** set when an ID is first published. If a publish computes an ID equal to the current `out/release.json` ID, it leaves the record, archive and `docs/progress.md` unchanged and reports `unchanged release <id>`. `release.json` is excluded from the byte-identity comparison of outputs.
- **Content:** snapshot ID/state and ExtractorVersion; model digest; extraction-code SHAs associated with runs and analysis-code SHA; manifest/candidate/PDF/run checksums or digest inventory; live-verdicts and theme-assignment checksums; guideline and recommendation counts; edition rule; validation gate; and checksums of published outputs. Record the search start and cutoff from `corpus/search.json`: the cutoff is not necessarily the day of publication. The analysis-code SHA is part of the ID, so a new commit produces a new release even when outputs are byte-identical.

**Archive:** `out/` is gitignored, so each new release is also archived under `artifacts/releases/<release-id>/` (local, gitignored) as `release.json` plus a consistent SQLite backup `evident.db` (backup API, §6). PDFs are not copied (size, licensing); their SHA-256 inventory is in the record, and the PDF directory belongs in the machine's regular backup. Keep every archive; no automatic pruning in M5. The git-tracked trace is a readable `docs/progress.md` entry naming the release ID. No external model or GPU is needed to rerender the same release once all verdicts are fixed.

## 6. Operator runbook (`docs/runbook.md`)

Write three complete procedures with actual `uv run` commands, prerequisites, expected output and recovery:

**A. Add a guideline:** backup DB → search and screen → edit candidate/manifest rows → download or import and verify PDF → `uv run python -m evident add <manifest-named-pdf>` → if adjudication is requested, run both judges for that DOI, merge to the live ledger, and `uv run python -m evident publish --snapshot <published-id>` → optionally assign themes for the new statements with `scripts/assign_themes.py`, or accept the **Not yet assigned** count that `status` reports → `uv run python -m evident status --published --check` → verify generated headline/dashboard and the release ID. Explain that the existing `add` automatically publishes when fully ready, and what exit code 3 means.

**B. Add a new edition:** perform A with a reviewed `supersedes_doi` link, then check the old DOI is historical-only, the new DOI is current-only, and both are represented in historical trends. Explain the exception for focused updates.

**C. Upgrade extractor:** backup DB → `snapshot --config ...` → `snapshot --resume` if needed → `validate --snapshot ID` → review duplicate verdicts → `publish --snapshot ID` → strict status and release archive. Document failed-gate handling with `reject --snapshot ID --reason TEXT`; do not use a regression override routinely.

Include a short troubleshooting section for changed PDFs (restore bytes, else `rebind-pdf`, else `add --force`), model digest mismatch, missing or inaccessible PDF, partial/failed run, duplicate verdict mismatch, unfinished candidate snapshots, and a failed render. Use SQLite's backup API or `sqlite3 .backup` to obtain a consistent backup; never just copy a live WAL-mode DB file. Serve the dashboard locally by default; public hosting requires a separate decision and approval.

## 7. Verification and acceptance tests

| # | Criterion | Test |
|---|---|---|
| 1 | Incremental add extracts only the new DOI, reuses the published version, and leaves other active runs unchanged | End-to-end fake-runner test with three existing PDFs plus one new PDF |
| 2 | Same DOI/same PDF is idempotent; modified PDF requires `--force`; hash clash with a different DOI is rejected | Pipeline, corpus and CLI fixtures |
| 2b | `rebind-pdf` accepts a PDF whose rebuilt prompts all match the run's stored hashes, keeping run, recommendations and verdicts and making no model call; it refuses a text change, a chunk-count change, a PDF without an active run, and a hash owned by another DOI, leaving the DB unchanged | Pipeline/CLI tests with a fake runner that fails if called; fixtures with metadata-only and text-changing PDF edits |
| 3 | Candidate/manifest mismatch, missing/changed PDF, or incomplete active-run coverage prevents **re-publish** and leaves existing output bytes unchanged | Failure-injection tests, including a PUBLISHED snapshot with a newly added manifest row |
| 4 | New screened duplicate pairs stop public refresh until adjudicated; two-judge consensus and grade safety match M4; stale text/ordinal verdicts are refused | Dedup/CLI integration tests using one new guideline and a forced replacement |
| 5 | M4's frozen DB, verdict SHA and archived package remain unchanged after a live add | File-hash assertions against `configs/paper1.json` and a live-ledger fixture |
| 6 | Whole-edition supersession affects current counts, not historical trends; a focused update can remain current beside its parent | Deterministic manifest/analytics fixtures |
| 7 | `status --published --check` sees published health when a newer candidate exists, fails for blockers, and treats triage warnings separately | Status/CLI tests |
| 8 | Upgrade builds one-version full snapshot, preserves old public outputs on failures, and applies the existing validation gate before switching | E2E snapshot/validate/publish/reject tests |
| 9 | A successful publication generates a reproducible release record and deterministic outputs; staged-render failure keeps the previous release | Publication tests with hash comparisons |
| 10 | Three runbook procedures are executable on a scratch DB; all existing tests, including the legacy golden replay, pass | Automated: one pytest per runbook procedure on a scratch DB with the fake runner and fake judges, run in `uv run pytest tests -m "not slow"`. Manual: procedure A is followed verbatim during the real demonstration below |

**Real M5 acceptance demonstration:** select **one genuinely eligible guideline published after the 2026-10-04 M4 search cutoff** (the later ACP menopause guideline is a candidate only after screening). On the live copy: curate it, acquire its PDF, ingest exactly that one PDF using v2 unless a newer version has already been published, review any new duplicate pairs, publish, and pass strict status. Verify the guideline appears in the current dashboard, the corpus/recommendation counts change as expected, and a second identical run is a no-op. Record commands, runtimes, source/artefact hashes, before/after counts, and the resulting release ID in `docs/progress.md`. Do not alter the M4 frozen source.

**If no eligible guideline exists yet:** M5 code is complete when tests 1–10 pass, but M5 exit waits for a real one. Re-run the search monthly with a new `--until`. Do not substitute a pre-cutoff guideline or an excluded item: the exit must prove a genuine post-freeze addition.

**M5 exit:** one post-MVP guideline successfully maintained end to end with **zero missing active runs, zero PDF drift, zero unresolved duplicate pairs, one extractor version per release, unchanged M4 archive, reproducible outputs, and a runbook that another operator can follow**.

## 8. Implementation order and decisions

1. **Protect the freeze:** separate `live_verdicts.csv`, pin the M4 file, and add text-sensitive verdict validation. Generalise the offline judge scripts beyond snapshot 8.
2. **Close the publish safety gaps:** candidate/manifest parity, coverage on republish, PDF hash checks with `rebind-pdf`, and fail-closed unresolved duplicate pairs. Stage output changes only after all preflight checks pass.
3. **Finish operator ergonomics:** `add --no-publish` for review staging, `status --published --check`, actionable errors and current release record. Keep ordinary successful `add` as a single command.
4. **Write tests and runbook** for routine add, new edition, upgrade and recovery. Preserve the M4/golden regression tests.
5. **Run the real demonstration, audit the dashboard, and document the result.** No public deployment or new extractor training is needed for M5 exit.

**Default choices for this draft:** unresolved duplicate pairs block final publication rather than being silently treated as distinct; M4 inputs are immutable; focused updates retain the earlier guideline unless the whole work is replaced; local dashboard hosting remains the default. Also: `add` exits 3 when duplicate review is pending; the release ID is a content digest, so an unchanged republish is a no-op; harmless PDF drift is accepted only by `rebind-pdf` when every rebuilt prompt hash matches; the search cutoff lives in `corpus/search.json`; releases are archived locally under `artifacts/releases/`. These are proposed M5 operational decisions and should be added to the design decision log when adopted.

## 9. Implementation notes (2026-10-26)

Where the code differs from, or settles, the text above:

- **Rebind provenance** goes to `corpus/pdf_rebinds.csv` (doi, run, old/new SHA-256, time), not a DB column: a schema migration would make the frozen M4 DB unreadable by `publish --package` (read-only opens require the code's schema version).
- **Exit 3** also applies to `publish` with unjudged pairs, not only `add`.
- **Dashboard** shows snapshot, version, model, code, search cutoff and the edition rule. The release id is in `out/release.json` only: it hashes the dashboard bytes, so the page cannot contain it.
- **`docs/progress.md`** is written by hand (runbook A.9); `publish` prints the release id and outcome.
- **Artefact gate** (§5.3) probes Ollama for the model digest on every live `publish` and `status --check`; Ollama must run, no model is loaded. The duplicate screen needs BioLORD, as before.
- **Strict status** also blocks when `out/release.json` is missing, names other active runs than the published snapshot's (e.g. after `add --no-publish`), or its output checksums no longer match `out/` (e.g. after `publish --package`, which renders the frozen M4 outputs there, or a crash between render and release write).
- **`rebind-pdf` with the recorded bytes** (e.g. from a backup) copies them back over a drifted PDF dir copy (`restored`).
- **Consensus** refuses two judge files whose texts differ for the same pair (a stale judge run after `add --force`).
- **PR #4 review (2026-10-27):**
  - Strict status also blocks when a release input changed since the release: manifest, candidates, search window, verdict ledger, themes, or the PDF hash of a released run (e.g. after `rebind-pdf`). Code commits alone do not count.
  - `publish` stages outputs, archives the record and DB backup, swaps `tables/`, `figures/`, `dashboard/` in all or none, then writes `out/release.json`. A failure leaves the previous outputs and record. After a candidate's transition, a failure leaves it published in the DB with the old outputs; strict status blocks until `publish --snapshot <id>`.
  - Every older edition in a topic must be named in a `supersedes_doi` (`;`-separated for several); a newer row no longer hides it implicitly. ERC-ESICM post-resuscitation 2025 now names temperature control 2022, which M4 folded in implicitly; counts unchanged (75 current, 10 superseded). The manifest SHA-256 no longer matches `configs/paper1.json`; the frozen manifest stays at tag `paper1-freeze`.
- **Known limitation:** the analysis-code SHA is `git rev-parse HEAD`; uncommitted code changes that leave the outputs byte-identical keep the release id. Marking a dirty tree would change the id on every curation edit before its commit. Commit code before a release.
- Live check on a copy of `out/evident.sqlite` (snapshot 8): 0 unresolved pairs with the seeded ledger (1256 screened, as in M4), 140 duplicates removed, release `f0e36aa1…`; a second publish was byte-identical and reported `unchanged`.

