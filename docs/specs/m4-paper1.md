# Spec: M4 — Paper Draft 1 (ICU + ACP)

**Status:** Proposed, 2026-10-13  
**Sources of truth:** `docs/roadmap.md` (M4), `docs/designs/living-evidence-map.md` (Decisions 1, 3–5, 8–10), `docs/specs/m3-mvp-snapshot.md`, `docs/progress.md` (M3 through 2026-10-13).

## 1. Goal

Produce a defensible, reproducible first-paper draft describing the strength, direction and certainty of **graded** recommendations in ESICM, SCCM and ACP clinical practice guidelines published from 2016 through the October 2026 corpus search. Every manuscript count, percentage, interval, table and figure must be derived from a **single frozen copy of the published M3 snapshot**, through `evident publish`. Do not manually transcribe numerical results into the manuscript.

M4 is a publication and validation milestone, **not another extractor-selection milestone**.

### Done at M3; reuse rather than rebuild

- Published **snapshot 8**, ExtractorVersion **v2** (`39ac17093b65…`, `configs/v2_qwen38-64k_alnum.json`), with 73 manifest guidelines and complete runs at M3; the pre-freeze search extension (2026-10-13) adds 12 guidelines, extracted into snapshot 8 with `add` (same v2 version): **85 guidelines, 75 current** (the 2026 NCS/SCCM ICH update supersedes 2016); historical analyses retain the superseded editions.
- Existing `evident/analytics.py` produces headline, certainty distribution, strength-by-certainty cross-tabulation and publication-year trends. `evident/publishing.py` writes six CSVs, four SVG/PNG figures, captions and a static HTML dashboard. Outputs have a byte-identical regeneration test.
- Twelve labelled ACP/ESICM guidelines (48 reference graded recommendations) give v2 graded F1 **0.939** (95% guideline-cluster bootstrap CI **0.867–0.989**), precision **0.885**, recall **1.000**; these labels also informed tuning, so this is **development-set performance**, not an external test.
- The M3 ten-guideline spot check found graded F1 **0.881** but low coverage of ungraded statements (16/160). Its reference lists (`out/spotcheck_m3/`, not yet committed) and the bias computation move into the package as an exploratory supplement table, since Decision 9 (graded-only) rests on it. Its graded headline bias was small on that sample, but it was single-reviewer and is **not** a replacement for an independent validation set.
- Three SCCM guidelines have already been preselected for held-out labelling; do not choose different or more favourable examples after examining results.

## 2. Scope and non-goals

**In scope:** a locked analysis dataset, journal-ready descriptive analyses and figure captions, per-guideline sensitivity results, guideline-cluster uncertainty intervals, separately reported held-out SCCM evaluation, full methods and limitations, a supplement and reproducibility manifest, and a draft manuscript whose numerical claims trace to `publish` artefacts.

**Out of scope:** extractor v3 or an ungraded-share result; cross-specialty expansion (M6); routine new-guideline ingestion (M5); recommendation-level supersession (M5); new predictive claims, significance tests or cross-society causal interpretations; and cross-society semantic-conflict matching. Discordance analytics from `TODOS.md` remain optional after the primary package is complete.

No retraining, new extraction or threshold tuning is allowed in the primary Paper 1 analysis. A genuine data/grade-mapping error must have a regression test, a documented correction and a regenerated, newly hashed analysis package. Preserve the original outputs for comparison.

## 3. Freeze the analysis source before drafting

1. Freeze a read-only **copy** of the snapshot 8 SQLite DB (not in git) and record its SHA-256; tag the repository commit holding the final `corpus/manifest.csv` and `corpus/candidates.csv` as `paper1-freeze` (git pins them; no copies). Input PDF SHA-256 hashes are already stored per run (`extraction_run.pdf_sha256`, 73/73). Do **not** use the evolving live `out/evident.sqlite` as a permanent paper source: `add` can append runs to a published snapshot.
2. Confirm the frozen source has one active successful v2 run for every included guideline (85/85), no unexplained exclusions, no PDF/hash drift, and that the edition resolver marks 75 current guidelines. Fail closed on a coverage mismatch, ambiguous edition link, mixed extractor version, or unresolved provenance.
3. Record the extractor version ID, full config JSON and prompt/example/model/embedding provenance already held by the version/run records; separate the **extraction code SHA** from the **analysis/publishing code SHA** (they need not be the same).
4. Keep the analysis as of the **2026-10-04 PubMed search**. Any guidelines added after the freeze belong to a later living-map update, not silently to this paper.
5. Make `publish` optionally write an analysis package that consumes the frozen DB and external held-out labels read-only; standard `publish` and the living dashboard continue to work without them. The option is generic (Paper 2 reuses it), not named after one paper.

Proposed CLI extension (illustrative interface):

```
git checkout paper1-freeze
python -m evident --db <frozen-db> \
  publish --snapshot 8 --package out/paper1 --heldout-labels <heldout-dir>
```

`--package` requires an explicit published snapshot ID. A missing, incomplete or mismatched held-out label set is an error, not a silent omission. No snapshot state, run, baseline gate or stored recommendation may be changed by the Paper 1 rendering path.

## 4. Corpus methods and flow

Inclusion rule (M3, clarified 2026-10-13): an ESICM-, SCCM- or ACP-led/co-led clinical practice guideline, in any journal; first published online from **2016-01-01 to the search date**; containing at least one explicitly strength- and certainty-graded recommendation. Include adult and paediatric guidelines and GRADE-based rapid guidelines where eligible. Name the excluded formats in Methods: ACP rapid living practice points (certainty stated, no strength), ACP guidance statements and best-practice advice, consensus documents, executive summaries and other derivative papers.

- Report the search date, exact PubMed queries from `scripts/search_guidelines.py`, the cross-check against the three society guideline web pages (done 2026-10-13 before the freeze; hits found only there carry `source=society_site`), first-online year convention and selection workflow. The current curation has **999 candidates** from three queries per society plus society pages, **85 included**; counts must be verified against the frozen CSVs.
- Emit a reproducible flow table from `candidates.csv`: included, and excluded by each enum reason (`not_guideline`, `not_graded`, `other_society`, `endorsement`, `derivative`, `duplicate_publication`, `out_of_range`, `pdf_unavailable`). List all excluded DOIs with reasons in the supplement, including none for a zero-count category.
- Count co-published/joint guidelines **once**, under the society of the retained publication; list collaborating societies separately. Report exclusions, PDF acquisition limits and the precise publication window (2026 is a partial calendar year).
- Retain all included editions in the archive. **Primary cross-sectional tables: current editions only. Historical trends: all editions.** Explicitly describe the manual `topic_id`/`supersedes_doi` rules, the 2022 ERC–ESICM temperature guideline moved to historical-only status, and the unresolved double counting of individual questions in focused updates (estimated at most ~25 graded recommendations).

## 5. Prespecified analysis and denominators

The unit of analysis is **one extracted, deduplicated recommendation**, not one guideline. Use the existing `HarmonizedRow` and `HarmonizedGrade`; never infer a missing grade. Primary tables include only `category == graded`.

**Within-guideline duplicates** (amendment 2026-10-24, fixed before any Paper 1 table was recomputed; `evident/dedup.py`). The extractor's own pass (alphanumeric key, BioLORD ≥ 0.97) leaves copies of one recommendation, e.g. a summary list repeated in the body or an abbreviation spelled out. A similarity threshold alone cannot separate them from sibling recommendations: on the 12 development guidelines the best BioLORD threshold reached pair F1 0.25, because "… in adults" vs "… in adults with kidney injury" scores higher than a hyphenation-noise copy (`scripts/paper1_dedup_calibration.py`). Rule:
1. Screen: every pair of rows of one guideline (all editions, graded or not) with BioLORD cosine ≥ 0.75 (CPU, the runs' embedding revision) **or** character ratio ≥ 0.70 on the normalized text (NFKC, lower case, trailing grade parenthesis removed, letters and digits only). A first screen at 0.85/0.85 (338 pairs) missed abbreviation copies ("NMBA" vs "neuromuscular-blocking agent": BioLORD 0.61–0.74).
2. Adjudicate: a local LLM labels each screened pair duplicate (same population, intervention, comparator and direction, restated) or distinct (`scripts/paper1_dedup_adjudicate.py`, temperature 0, fixed seed, guideline title as context). Verdicts are frozen in `corpus/dedup/paper1_verdicts.csv`, hashed in `configs/paper1.json`; `publish` performs no LLM inference and fails closed when a screened pair has no verdict.
   - Prompt development: the first prompt agreed with an independent reviewer on 313/338 first-screen pairs (3 false merges, 22 missed duplicates: dropped scope words, appended remarks, truncation); the revised prompt names those cases. Its agreement is reported on a random sample of pairs outside the first screen, labelled blind to the LLM verdict.
3. Duplicates form clusters (connected components). Kept: the copy whose strength and certainty are both mapped; then the lowest ordinal. Clusters whose copies disagree on the harmonized grade are counted and reported, not resolved.
4. Every count, table, figure and both validation scores use the deduplicated rows. The pre-dedup package (2026-10-20) is preserved for comparison.
5. Amendment 2026-10-25, after the first deduplicated package was seen: a human can check about 15 pairs, too few to estimate verdict accuracy (0/15 errors still allows 22%). Hence:
   - **Judges:** two local models from different families judge every screened pair with the step 2 prompt (`qwen38-64k`, `gemma3:27b`); where they disagree, a third (`mistral-small3.2:24b`) decides. Per-judge files in `corpus/dedup/judges/`; `scripts/paper1_dedup_consensus.py` writes the frozen verdicts.
   - **Bound:** Table S2 recomputes Table 1 with every copy kept; any wrong merge moves Table 1 toward it.
   - **Human check:** 15 pairs, fixed seed (`scripts/paper1_dedup_human_sample.py`): 5 merges whose copies differ in grade, 5 merges with the same grade, 5 pairs the first two judges split. Reported as errors per stratum with an exact CI, as a check, not a corpus error rate.

| Outcome | Numerator | Denominator |
|---|---|---|
| Strong (%) | `strength == strong`, regardless of direction | Current graded recommendations with **mapped strength** |
| Against (%) | `direction == against`, strong or conditional | Current graded recommendations with **mapped strength/direction** |
| High/moderate certainty (%) | `certainty in {high, moderate}` | Current graded recommendations with **mapped certainty** |
| Certainty distribution (%) | Each of high, moderate, low, very low | Current graded recommendations with **mapped certainty** |
| Strength × certainty | Counts for strong/conditional × four certainty levels | Current graded recommendations with **both axes mapped**; row percentages use the corresponding mapped-strength row |

Show numerator **and** denominator beside every percentage, and separate excluded/unmapped/unknown counts. An empty denominator renders as `NA`, never as zero. Calculations use unrounded values; CSVs/figures display one decimal place. 'Strong' includes strong against recommendations; a statement offering **no direction** is `no_recommendation`, not a graded against recommendation, irrespective of the model's guessed label.

**Do not report the v2 ungraded share as a Paper 1 finding.** The existing dashboard's ungraded fields must be hidden from the publication view or explicitly flagged as incomplete; unchanged exploratory exports may retain the columns for compatibility.

### Primary results

1. **Table 1, society-level summary:** ACP, ESICM, SCCM and ALL; current guideline count, total extracted graded count, mapped denominators, strong %, against %, high/moderate % and 95% CIs. ALL is pooled **per recommendation** and labelled as SCCM-dominated, not the mean of society percentages.
2. **Table 2, certainty and strength distribution:** certainty counts/percentages and mapped strength-by-certainty counts, by society and ALL. Reuse `certainty_distribution` and `strength_by_certainty`.
3. **Figures:** reuse the four deterministic publication figures (`fig1` certainty, `fig2` strong, `fig3` cross-tabulation, `fig4` trends). Ensure that against-recommendation rates are visible in Table 1; add a figure only if the manuscript requires it. Each figure gets a short, distinct caption, not M3's repeated full provenance paragraph.
4. **Time trends:** by first-online year and society, **all editions**, with guideline count and graded denominator per point. Flag 2026 as incomplete and small year-by-society cells as descriptive; do not assert temporal change solely from plotted slopes. Any additional grouping (e.g. two-year bins) must be specified before looking at the pattern and labelled supplementary.
5. **Supplementary per-guideline sensitivity table (Decision 10):** for each society and ALL, the unweighted mean of each individual guideline's mapped recommendation percentage, the number of contributing guidelines, and the difference in percentage points from the pooled primary estimate. Use current editions; omit zero-denominator guidelines from that axis and disclose their count. Never average already-rounded percentages.

### Uncertainty

Extend the existing guideline-cluster bootstrap approach to Paper 1 headline proportions, rather than treating recommendations as independent observations. Default: **2,000 resamples, fixed RNG seed 0, 2.5th/97.5th percentiles**, operating on per-guideline numerator/denominator pairs. Resample guidelines **within society** for society CIs; for ALL, stratify draws by society and recompute the pooled per-recommendation ratio. Preserve the same estimator/denominator in every draw; drop undefined draws and disclose if any exist. Report that these intervals describe between-guideline variability in this curated corpus and **do not correct extraction bias or establish population representativeness**.

## 6. Held-out SCCM validation (separate from the gate)

The preselected guidelines are:

| Guideline | DOI |
|---|---|
| Safe medication use (2017) | `10.1097/ccm.0000000000002533` |
| Heat stroke (2025) | `10.1097/ccm.0000000000006551` |
| Paediatric end-of-life care (2026) | `10.1097/pcc.0000000000003914` |

They were drawn with seed 1 from the 27 current SCCM guidelines not used in the M3 spot check. The existing blind templates live outside the 12-guideline label directory, under `<data root>/intensive_care_medicine_heldout/`; retain that separation.

- Complete full-PDF reference annotations **blind to v2 predictions**, using the established ICU spreadsheet schema plus page and annotation rules. Label all eligible graded recommendations, along with best-practice/no-recommendation statements for later v3 work. Check DOI/edition identity, duplicate rows, grades and page traceability; resolve ambiguous units of recommendation before scoring.
- Score the frozen **stored** v2 runs with the existing `validation.score` matching and harmonization rules. Reuse 0.65 BioLORD/Hungarian matching, TP/FP/FN, micro precision/recall/F1, and strength-with-direction and certainty accuracy on matched graded pairs (as `validate` scores them). Report per-guideline TP/FP/FN and the pooled micro P/R/F1 with an explicit **n=3, imprecise** qualification. No guideline-cluster bootstrap: 3 clusters give 10 distinct resamples. If an interval is shown, use a Wilson interval on recommendations, labelled as ignoring clustering.
- Optional: a second annotator labels one of the three guidelines; report agreement.
- Publish the ungraded diagnostics separately in a supplement, clearly marked exploratory; do not add an ungraded-share estimate to the paper.
- **Do not feed these labels into** `validation.load_labelled()`, few-shot selection, extractor tuning, the existing 12-guideline gate, or any model revision. Do not present the M3 spot-check scores as independent held-out validation.
- Keep the existing `evident validate` gate, including its original 12 labels and baseline comparison, unchanged. The new SCCM test is a **read-only Paper 1 report**, not a new publication gate.
- Change the dashboard/caption language from an unqualified 'SCCM accuracy unvalidated' to 'SCCM evaluated separately on 3 held-out guidelines; not part of the 12-guideline development gate', **only after** those results exist. Keep the cohort-size limitation prominent.

## 7. `publish` outputs and implementation boundaries

Extend `PublishBundle` with a typed optional Paper 1 payload, populated in the **pipeline read path** from the frozen store, manifest, candidates and SCCM reference labels. Put transformations and bootstrap calculations in pure functions in `evident/analytics.py` or a small `evident/paper1.py`; leave `evident/publishing.py` responsible only for deterministic rendering. The external labels loader must be explicit and validated. No direct SQL in new scripts, and no model inference in `publish`.

Retain current `out/tables/`, `out/figures/` and `out/dashboard/` outputs. With `--package` it stages and atomically swaps an additional subtree:

```
out/paper1/
  table1_society.csv
  table2_certainty_strength.csv
  table_s1_guideline_mean.csv
  figure_data.csv                 # point estimates, numerator, denominator, CIs
  corpus_flow.csv
  excluded_guidelines.csv
  raw_harmonized_mapping.csv
  validation_development.json    # existing 12-guideline report, labelled as in-sample
  validation_sccm_heldout.json   # three new guidelines, no gate
  spotcheck_bias.csv             # M3 spot check: published vs correct vs reference headline (exploratory)
  draft.md                       # rendered manuscript
  captions.md
  methods_facts.json             # counts and manuscript-ready provenance fields
  analysis_manifest.json        # source hashes, parameters, output hashes
```

The published SVG/PNG figures may stay in `out/figures/`; `out/paper1/figure_data.csv` must suffice to reconstruct every plotted point. `analysis_manifest.json` must include snapshot ID and state, full extractor ID/model digest, extraction and analysis Git SHAs, corpus/search cutoff, manifest/candidates/DB checksums, held-out label checksums, edition rule, publication bootstrap seed/count, analysis config and hashes of all emitted Paper 1 files. Avoid a manifest hashing itself: exclude `analysis_manifest.json` from its own output hash inventory.

All manuscript numbers, including corpus flow and both validation summaries, come **only** from these exports. The draft is a template (`docs/paper1/draft.md.j2`) referencing exported fields, e.g. `{{ table1_society.ACP.pct_strong }}`; `publish --package` renders it to `out/paper1/draft.md`. An unknown field fails the render (fail closed); no number is typed by hand. The two sets of validation results must never be silently merged.

Run the deferred `/plan-design-review` on the paper figures and dashboard, fix the repeated caption problem, and keep the dashboard local-only (`127.0.0.1`) by default unless the operator explicitly decides to expose it on the LAN. Design changes must not alter computations.

## 8. Manuscript and supplement

Create `docs/paper1/draft.md.j2` (rendered by `publish --package`; a journal template later) with the following sections:

- **Title/abstract:** an automated descriptive analysis of graded recommendations, explicitly limited to the three societies and publication window.
- **Introduction:** motivation, distinction between recommendation strength and certainty, and the role of an updateable extraction pipeline.
- **Methods:** search/query/date and screening; eligibility/exclusions and duplicate publications; PDF sources; v2 model/strategy/config/version; recommendation unit, GRADE/ACP mapping and raw-grade retention; category and denominator rules; edition treatment; primary and sensitivity estimators; bootstrap specification; 12-guideline development validation and three-guideline independent SCCM test; analysis provenance.
- **Results:** flow diagram/counts, Table 1 and Table 2, trend and strength/certainty figures, per-guideline sensitivity results and validation summaries. Values are imported from the frozen `publish` package.
- **Discussion:** comparison with Fanaroff et al. (JAMA 2019, ACC/AHA cardiology) and earlier critical care level-of-evidence studies; differences among societies as descriptive findings; development-set optimism, three-SCCM-guideline sample size, imperfect precision, one-reviewer M3 spot check, variable PDF layouts, incomplete ungraded capture, remaining grade mapping uncertainty, focused-update overlap, curated corpus/search limits and partial 2026.
- **Conclusion:** restricted to observed graded-recommendation distributions and feasibility/reproducibility; no claim that a higher strong/low-certainty rate proves inappropriate care.

The supplement contains the complete DOI manifest, exclusion list with one reason per candidate, queries/search date/selection flow, complete observed raw-to-harmonized label mapping with rules and counts, co-publication/supersession list, per-guideline sensitivity results, per-guideline SCCM validation and annotation protocol, full figure data, source/code/checksum manifest, and an executable reproduction command. Share PDFs only where licensing permits; hashes and bibliographic IDs suffice for provenance.

## 9. Verification and exit criteria

| # | Criterion | Verification |
|---|---|---|
| 1 | Frozen source is coherent; 85 active v2 runs, 75 current editions, no hidden exclusions or altered PDFs | Frozen-DB audit; repository manifest/candidate checks; `status` and source-hash inventory |
| 2 | Paper 1 uses graded rows only with correct axis-specific denominators | Unit fixtures: unmapped, ungraded, no-direction, strong-against, empty group |
| 3 | Superseded editions are absent from cross-sectional tables but included in trends; joint copies count once | Deterministic fixture with an old/current topic and a co-publication |
| 4 | Pooled vs guideline-mean sensitivity differs where guideline sizes differ; zero denominators handled | Two unequal-size guideline fixture with known hand-calculated results |
| 5 | Cluster CIs are deterministic, computed at guideline rather than recommendation level, and society-stratified for ALL | Fixed-seed fixtures, tiny clusters, undefined denominator tests |
| 6 | SCCM labels cover exactly the three prespecified DOIs; no contamination of tuning labels/gate | Strict label-loader tests and identical baseline 12-guideline report before/after |
| 7 | Paper exports, captions and dashboard are generated by the same `publish` path and remain intact if rendering fails | `test_publishing.py` / `test_e2e.py`; failure-injection and byte-for-byte rerender test |
| 8 | Every primary and supplementary numerical manuscript claim traces to a frozen exported field and validation sample is clearly identified | Audit draft against `out/paper1/` with no unexplained literals |
| 9 | All existing tests, legacy golden replay and new tests pass | `python -m pytest tests -m "not slow"` plus relevant integration tests |
| 10 | A complete first-paper draft and supplement exist; one command reproduces all tables and figures from stored runs without an LLM/GPU | Reproduce on the frozen inputs, compare output hashes, archive log and manifest |

**M4 exit:** a complete reviewable draft, generated Paper 1 tables/figures/supplement and a reproducibility record, all tied to one immutable snapshot 8 source; the development and held-out validation results are unambiguously separated. This is **not** an assertion of journal submission readiness or external generalisability.

## 10. Implementation order and open decisions

1. **Freeze/audit first:** source copy, hashes, edition resolution, exclusion flow, publication variables. No paper numbers before this passes.
2. **Pure analytics:** mapped denominators, Table 1/2, guideline-mean sensitivity, cluster bootstrap and tests.
3. **Independent SCCM labels:** finish blind annotation; implement read-only scoring/reports and tests. May run in parallel with step 2.
4. **Publisher:** `publish --package`, staged outputs and `analysis_manifest.json`; regenerate/reconcile existing figures and dashboard language; run design review.
5. **Writing:** assemble manuscript/supplement from named generated fields; conduct a numeric and methodological audit; lock the reproducibility archive.

**Effort (rough):** 4–6 days of code and writing, plus the held-out labelling time.

**Decide before implementation:** journal/target reporting format and whether to display descriptive bootstrap CIs in Table 1 or only the supplement. Defaults: journal-neutral Markdown draft; include the CIs in Table 1, qualified as within-corpus guideline variability. Keep dashboard access local-only unless separately approved.

**Optional only after exit:** discordance audit (strong recommendations with low/very-low certainty; no cross-society matching without a separately validated threshold). It must not change the primary results or delay the paper.

