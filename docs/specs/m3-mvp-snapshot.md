# Spec: M3 MVP Snapshot

Status: DRAFT · 2026-10-04 · Branch: `agent_explor`
Parent docs: `docs/designs/living-evidence-map.md` (decisions), `docs/roadmap.md` (M3), `docs/specs/m1-pipeline.md` (pipeline, CLI, T11, T8a)
Tasks: corpus search + manifest · PDF acquisition (T11 run) · v1 corpus snapshot · v1 spot check · T8 analytics + publishing + dashboard

## 1. Goal

Run v1 (design Decision 7, `configs/v0_qwen38-64k_dedup097.json`) over every
ESICM, SCCM and ACP guideline published 2016–2026, publish it, and turn
`publish` into the single producer of tables, figures, captions and a static
dashboard. M3 ends with the first real evidence map: one published snapshot
of the full MVP corpus, every number regenerated from the DB.

```
 PubMed search ─▶ candidates.csv ─(human include/exclude)─▶ manifest.csv
                                                               │
 fetch_pdfs.py (VPN for SCCM) ─▶ PDF dir ◀─ manual download ───┘
        │
        ▼
 snapshot (v1) ─▶ status triage ─▶ validate (vs snap 5) ─▶ publish
                                                             │
        ┌────────────────────────────────────────────────────┘
        ▼
 out/tables/*.csv  out/figures/*.svg|png + captions.md  out/dashboard/index.html
        │
        ▼
 spot check of v1 on unlabelled guidelines → docs/progress.md
```

### Non-goals (M3)
- Discordance analytics (TODOS.md): after M3, optional in M4.
- Paper text, Fanaroff tables in final form, cluster-bootstrap CIs on headline percentages: M4.
- New ground truth (design Decision 3). The spot check (§7) records counts, not GT rows.
- Table-first (T5), FT re-evaluation (T9), any new ExtractorVersion. A v1 failure on the corpus is a finding; it reopens M2, not M3.
- DOI input to `add`, BioLORD topic matching, hosting PDFs or full text.

## 2. Constraints

| Constraint | Consequence |
|------------|-------------|
| Python 3.9, no new dependencies | matplotlib 3.9.4 and jinja2 3.1.6 are installed; `urllib` for PubMed. No plotly, no CDN in the dashboard |
| User rules (global CLAUDE.md) | enums not booleans, private by default, early return, constants, minimal diff in existing modules |
| Layering | `cli → pipeline → {corpus, analytics, publishing, store, …}`. `analytics` is pure (rows in, aggregates out). `publishing` renders aggregates; it never reads the store nor imports `harmonization` |
| One code path (Decision 4) | figures, tables, captions and dashboard come from the same aggregates in one `publish` call |
| Determinism | `publish` twice on the same DB gives byte-identical outputs (no timestamps, fixed SVG hash salt, sorted rows) |
| Copyright (design NOT in scope) | dashboard shows recommendation text, page and DOI link only; no PDF, no full text |
| Sci-Hub | the search script and T11 never import `extraction.pdf_loader` (M1 §2) |

## 3. Layout

```
evident/
  analytics.py     + distribution, trends, crosstab, caption facts (pure)
  publishing.py    NEW: CSV tables, matplotlib figures, captions, dashboard HTML
  templates/dashboard.html.j2   NEW
  corpus.py        + candidates file loading and cross-check
  pipeline.py      _write_outputs → analytics → publishing; PublishedRec rows
scripts/
  search_guidelines.py   NEW: PubMed search → candidates draft
corpus/
  candidates.csv   NEW: every search hit with a decision and reason
  manifest.csv     12 rows → full MVP corpus
tests/evident/
  test_analytics.py  test_publishing.py (new)  test_corpus.py  test_pipeline.py  test_e2e.py
tests/scripts/
  test_search_guidelines.py (new)
out/
  tables/  figures/  dashboard/          (generated, git-ignored)
```

## 4. Corpus

### 4.1 Inclusion rule (goes verbatim into the M4 methods)
Include a publication when all hold:
1. a clinical practice guideline led or co-led by ESICM, SCCM or ACP;
2. published (first online) 2016-01-01 to the search date;
3. it states at least one recommendation with a strength and a certainty grade (GRADE or ACP's GRADE-based system).

Exclude, with one reason each (`ExclusionReason` enum, §4.3):

| Reason | Example |
|--------|---------|
| `NOT_GUIDELINE` | narrative review, research agenda, methods paper |
| `NOT_GRADED` | consensus statement, Delphi, ACP "best practice advice" or "guidance statement" |
| `OTHER_SOCIETY` | published in an MVP journal but led by another body (ESPNIC, Brain Trauma Foundation, British societies) |
| `ENDORSEMENT` | ACP/SCCM endorsement of another body's guideline |
| `DERIVATIVE` | summary for patients, "In the Clinic", executive summary of an included guideline |
| `DUPLICATE_PUBLICATION` | the second journal copy of a co-published guideline (e.g. SSC 2021 in CCM and ICM) |
| `OUT_OF_RANGE` | first online before 2016 |
| `PDF_UNAVAILABLE` | no PDF after T11 and a manual attempt (§5); last resort, reported in the supplement |

### 4.2 Search (`scripts/search_guidelines.py`)
```
python scripts/search_guidelines.py --email YOU@ORG [--out corpus/candidates_draft.csv]
```
- PubMed E-utilities (`esearch` + `esummary`), stdlib `urllib`, 0.4 s between requests (NCBI limit without key).
- One query per society, held as module constants so the methods can quote them:

  | Society | Journals | Title filter |
  |---------|----------|--------------|
  | ESICM | Intensive Care Med | guideline* |
  | SCCM | Crit Care Med, Pediatr Crit Care Med | guideline* |
  | ACP | Ann Intern Med | "American College of Physicians" AND guideline* |

  Date: `2016/01/01:<search date>[dp]`. Exact strings are tuned during implementation and frozen in the constants.
- Output columns: `doi, society, year, title, journal, pmid, source, decision, reason, duplicate_of`. `decision` and `reason` are left empty for the human.
- Cross-check against the three society guideline web pages by hand; hits found only there get `source=society_site`. Search date, query strings and hit counts go into `docs/progress.md`.
- Idempotent merge: re-running keeps existing decisions, appends new hits.
- Tests (mocked `_http_get`): query strings; esummary parsing (DOI from `articleids`, year from `epubdate` else `pubdate`); merge keeps decisions; email only in query params.

### 4.3 Candidates file (`corpus/candidates.csv`)
`corpus.load_candidates(path) -> list[Candidate]`, validated like the manifest (collect all problems, then raise `ManifestError`):
- `decision ∈ {include, exclude}`; an exclude needs a `reason` from `ExclusionReason`; `DUPLICATE_PUBLICATION` needs `duplicate_of` naming an included DOI.
- `corpus.check_candidates(candidates, manifest)`: every `include` is a manifest row and every manifest row is an `include`. Mismatch → `ManifestError` listing both sides.
- The pipeline does not read candidates; it is a corpus-curation artefact for the supplement. A test runs `check_candidates` on the repo files.

### 4.4 Manifest rows
- One row per included guideline; existing schema (`doi, society, year, title, topic_id, supersedes_doi, grading_family, pdf_filename`).
- New optional column `joint_with`: other MVP societies co-leading the guideline (`;`-separated, e.g. `ESICM` on the SSC row). `society` is the society of the journal copy kept. Analytics counts the row once, under `society`; captions name joint guidelines.
- `year` = first online year (same as the 12 existing rows).
- `topic_id` by hand; living ACP guidelines and SSC editions chain through `supersedes_doi`.
- `test_repo_manifest_is_valid` asserts the new count and that `check_candidates` passes.

Estimate: ~90–110 guidelines (≈20 ESICM, ≈30 SCCM, ≈45 ACP).

## 5. PDF acquisition

1. Outside the VPN: `python scripts/fetch_pdfs.py --email …` (ESICM and ACP are mostly open access).
2. Inside the VPN: same command on the machine with journal access, then copy the new PDFs into `/mnt/data1/klug/datasets/evidence_extraction/pdfs`.
3. Every remaining `PAYWALLED` / `NOT_FOUND` / `INVALID_PDF` line: download by hand into the PDF dir with the manifest filename, or move the guideline to `PDF_UNAVAILABLE` in candidates and drop its manifest row.
4. Done when a final fetch report is all `PRESENT`. Commit the report summary (counts by status and society) to `docs/progress.md`, not the PDFs.

No code change expected. A T11 bug found here follows the bug rule (test first).

## 6. Snapshot run

```
python -m evident snapshot --config configs/v0_qwen38-64k_dedup097.json   # est. ~1 h for ~100
python -m evident status
python -m evident snapshot --resume          # after fixing failures
python -m evident validate                   # baseline: snapshot 5
python -m evident publish
```

### 6.1 Triage before validate
For each item `status` reports:

| Item | Action |
|------|--------|
| FAILED run | read the error; fix the cause; `--resume` |
| FAILED_EMPTY | scanned PDF or no rec layout: check the PDF; exclude with reason only if no text layer |
| slow (> 3× median) | expected for long guidelines (SSC ~90 recs); note it |
| low-recall / ungraded-share outlier | open the PDF; a real miss goes into the spot check findings (§7) |

`--exclude` is allowed only with a reason that names the cause; each exclusion is reported in captions.

### 6.2 Reproducibility check (OV7)
The 12 labelled guidelines are re-extracted with the same version as snapshot 5. Expected: identical per-guideline TP/FP/FN, gate PASS. Any difference is logged and explained before `publish` (seed, Ollama version, or GPU nondeterminism).

### 6.3 Harmonization coverage
Harmonization runs on read, so a mapping fix needs no re-extraction. After the snapshot:
- list raw strength / certainty labels that come out `UNMAPPED`, by society;
- if unmapped exceeds 2% of graded recs in a society, extend `harmonization.py` with table tests from the real phrasing (bug rule: test first), then re-publish.

## 7. v1 spot check on unlabelled guidelines

Every v1 choice was made on the 12 guidelines it is scored on (progress.md, M2). The spot check gives a first out-of-sample read, SCCM included.

- Sample: 3 current-edition guidelines per society not in the labelled set, drawn with `numpy.random.default_rng(0)` from the DOI-sorted list; SSC 2021 is added to SCCM by hand (largest, table-formatted).
- Per guideline, against the PDF's own recommendation list/boxes: TP, FP, FN (graded only), strength and certainty correct on TPs, and the cause of each error (dedup, rationale restatement, table, other).
- Recorded in `docs/progress.md` as a table; no GT file is written.
- Reading: precision and recall well below the labelled 0.87 / 1.00 on SCCM is reported as a finding and becomes an M4 limitation or reopens M2. It does not block M3's exit.

## 8. T8 Analytics (`evident/analytics.py`, pure)

### 8.1 Input rows
`HarmonizedRow` gains the fields the dashboard and trends need; the pipeline builds it from active runs of **all** editions:
```python
@dataclass(frozen=True)
class HarmonizedRow:
    doi: str
    society: str
    year: int
    grade: HarmonizedGrade
    is_current: CurrentEdition        # enum CURRENT / SUPERSEDED; headline filters on CURRENT
    text: str
    page: Optional[int]
    raw_strength: str
    raw_certainty: str
```
Guideline-level facts (title, topic, joint_with, excluded reason) travel as `GuidelineFacts` rows built from the manifest and members.

### 8.2 Aggregates
| Function | Editions | Output |
|----------|----------|--------|
| `headline(rows, …)` | current | unchanged (M1 §12) |
| `certainty_distribution(rows)` | current | per society + ALL: n and % HIGH / MODERATE / LOW / VERY_LOW over graded with certainty MAPPED |
| `strength_by_certainty(rows)` | current | per society + ALL: crosstab strength (strong/weak) × certainty, counts and row % |
| `trends(rows)` | all | per society × year: n guidelines, n recs, n graded, % strong, % high/moderate; empty cell when the graded denominator is 0 |
| `caption_facts(rows, guidelines, labelled_dois)` | — | per figure: n guidelines, n recs, n excluded members (+reasons), n unmapped per axis, n superseded editions, joint guidelines, **unvalidated societies** |

- Unvalidated societies are computed, not hard-coded: societies in the corpus with no guideline in the labelled set. Today: SCCM.
- Denominators follow the design rule (graded only for %; ungraded share separate). One-decimal rounding, empty never 0 (M1 §12).

### 8.3 Tests
Hand-built rows per function: denominators exclude unmapped and ungraded; trends keep superseded editions, headline drops them; crosstab sums; unvalidated society derived from labelled DOIs; ALL equals the sum of societies.

## 9. T8 Publishing (`evident/publishing.py`)

### 9.1 API
```python
@dataclass(frozen=True)
class PublishBundle:            # everything one publish renders; built by the pipeline
    snapshot_id: int
    extractor_version_id: str
    model: str
    code_sha: Optional[str]
    validation: Optional[GateReport]   # latest validation of the snapshot
    headline: list[HeadlineRow]
    certainty: list[...]; crosstab: list[...]; trends: list[...]
    captions: CaptionFacts
    guidelines: list[GuidelineFacts]
    recs: list[HarmonizedRow]          # current and superseded, for the browser

def render(bundle: PublishBundle, out_dir: str) -> list[str]   # paths written
```
Writes into a temp dir next to `out_dir`, then swaps it in (no half-published state).

### 9.2 Outputs
```
out/
  tables/headline.csv  certainty.csv  strength_by_certainty.csv  trends.csv  guidelines.csv  recommendations.csv
  figures/
    fig1_certainty_by_society.{svg,png}      stacked bars, % of graded
    fig2_strong_by_society.{svg,png}         % strong (for + against), n on bars
    fig3_strength_by_certainty.{svg,png}     crosstab heatmap, counts
    fig4_trends.{svg,png}                    % strong and % high/moderate by year, one line per society, n per point
    captions.md                              one caption per figure and table, from caption_facts
  dashboard/
    index.html                               self-contained: inline CSS/JS, figures as inline SVG, data as embedded JSON
```
- `out/headline.csv` moves to `out/tables/headline.csv`; `PipelineDeps.headline_path` becomes `publish_dir`.
- Captions state: corpus, editions used, n guidelines/recs, denominator, excluded and unmapped counts, joint guidelines, "SCCM: extraction accuracy unvalidated (no labelled SCCM guideline)", snapshot id, version id, model label.
- Figures: matplotlib `Agg`, `svg.hashsalt` fixed, `metadata={"Date": None}`; PNG at 300 dpi for the paper. Colour per society fixed in one constant map.

### 9.3 Dashboard (content; layout from `/plan-design-review`)
1. Provenance bar: snapshot, version, model, code SHA, validation (F1, combined accuracy, 95% CIs, gate).
2. Unvalidated banner for each unvalidated society.
3. Headline table and the four figures with captions.
4. Guideline table: title, society, year, current/superseded, recs, ungraded share, DOI link.
5. Recommendation browser: filter by society, year, strength, certainty, category; text search; shows text, page, raw and harmonized grade, guideline. Vanilla JS over the embedded JSON.

Works from `file://` and from a static server. No external requests (test asserts no `http` URL except DOI links).

### 9.4 Serving
Static files, no server code:
```
python -m http.server 8000 --directory out/dashboard --bind 127.0.0.1
```
Run under a user systemd unit or tmux. Local-only vs LAN is open question 1 (§13); LAN = `--bind 0.0.0.0` behind the host firewall.

### 9.5 Tests (`test_publishing.py`)
- `render` on a small bundle writes every listed file; nothing outside `out_dir`.
- Two renders → byte-identical files.
- Captions contain the counts and the unvalidated sentence; no unvalidated sentence when every society is labelled.
- HTML: embedded JSON parses; row count equals `recs`; no external URLs except `https://doi.org/`.
- A failure mid-render leaves the previous `out_dir` intact.

## 10. Pipeline changes (`evident/pipeline.py`)
- `_write_outputs`: build `HarmonizedRow` for all editions and `GuidelineFacts`; call analytics; fetch the latest validation; build `PublishBundle`; `publishing.render`. `PublishReport` lists the written paths.
- `labelled_dois` for caption facts comes from `deps.labelled()` (already a dep).
- `add` keeps chaining `publish`, so one added guideline regenerates every output (design success criterion 1).
- Tests: publish writes all outputs for the published snapshot; superseded editions appear in trends and the browser, not the headline; re-publish after `add` changes counts.
- `test_e2e.py`: assert `out/tables`, `out/figures`, `out/dashboard/index.html` after each publish.

## 11. Exit criteria and verification

| # | Criterion | Check |
|---|-----------|-------|
| 1 | All tests pass, legacy golden unchanged | `python -m pytest tests -m "not slow"` |
| 2 | Every included candidate is in the manifest and vice versa | `test_corpus.test_repo_candidates_match_manifest` |
| 3 | Final fetch report all `PRESENT` | `fetch_pdfs.py` report |
| 4 | Snapshot PUBLISHED; `status` shows no coverage gap; every exclusion has a reason | `python -m evident status` |
| 5 | Labelled 12 reproduce snapshot 5 (or the difference is explained) | `validate` report, §6.2 |
| 6 | Every output regenerated from the DB: `rm -r out/tables out/figures out/dashboard; python -m evident publish` recreates byte-identical files | shell + `sha256sum` |
| 7 | Dashboard reachable | `curl -sf http://127.0.0.1:8000/ >/dev/null` |
| 8 | Spot check and corpus/run results recorded | `docs/progress.md` |

## 12. Implementation order

```
Lane A (corpus):     search script (§4.2) ─▶ candidates + manifest (§4.3–4.4, human) ─▶ fetch (§5, VPN)
Lane B (evident/):   analytics (§8) ─▶ /plan-design-review ─▶ publishing (§9) ─▶ pipeline (§10) ─▶ e2e
Join:                snapshot + triage (§6) ─▶ validate ─▶ publish ─▶ spot check (§7) ─▶ serve (§9.4)
```
Lanes A and B are independent; A is mostly human time (curation, VPN). One commit per numbered section.

## 13. Open questions
1. **Dashboard access:** local-only (`127.0.0.1`) or LAN? Default local-only.
2. **Pediatric guidelines** (SCCM/ESICM pediatric SSC, PCCM): in the MVP corpus? Default include; they are society-led and GRADE-graded.
3. **ESICM Rapid Practice Guidelines:** default include (GRADE, society-led).

## 14. Decisions taken in this spec (flag if you disagree)
1. **PubMed search + society web pages, logged in `candidates.csv`.** Gives a quotable search strategy and the supplement's exclusion list.
2. **Co-published guidelines count once**, under the society of the kept journal copy, with `joint_with` naming the others.
3. **`PDF_UNAVAILABLE` removes a guideline from the manifest** instead of a snapshot exclusion, so the published snapshot has no PDF gaps; it stays visible in the supplement.
4. **Same version as snapshot 5.** No config change in M3; the labelled 12 double as a reproducibility check.
5. **Spot check records counts only**, no GT file, to respect Decision 3.
6. **matplotlib + jinja2, self-contained HTML.** Same SVGs in paper and dashboard (Decision 4); no new dependency, works offline.
7. **Headline CIs move to M4.** M3 shows validation CIs only.
8. **`out/headline.csv` moves to `out/tables/`**; `headline_path` becomes `publish_dir`.
9. **Harmonization gaps are fixed on read**, without a new snapshot (§6.3).

## 15. Risks
- **SCCM accuracy is unknown.** No labelled SCCM guideline; SSC-style rec tables are a layout v1 has not been scored on. The spot check is the only evidence; captions say so.
- **Long guidelines** (SSC ~90 recs over 30+ pages): dedup at 0.97 was tuned on short ones; near-duplicate restatements may return as FPs.
- **Corpus size and SCCM access.** Paywalled SCCM PDFs depend on the VPN machine; missing PDFs shrink SCCM, the society with least validation.
- **`status` low-recall rule** misfires on small guidelines (M1 run). Triage costs time on ~100 guidelines; tune `_LOW_RECALL_FACTOR` only if the noise is unworkable.
- **Year binning:** per-year trends have few guidelines per society; small denominators are shown with n, not hidden. Binning is an M4 choice.
- **Test collection is slow** (TODOS.md); new tests add to it.
