# Implementation Roadmap: Living Evidence Map

Source of truth for decisions: `docs/designs/living-evidence-map.md`.
Task IDs (T1–T11) refer to that doc. Deferred items: `TODOS.md`.

## Overview

```
 M0 Foundations ──▶ M1 Pipeline ──▶ M2 Extractor v1 ──▶ M3 MVP snapshot ──▶ M4 Paper 1
   (T1 T2 T3 T6)     (T4 T7 T11)     (T5 T9 T10)          (T8, corpus)        (ICU+ACP)
                                                                 │
                                                                 ▼
                                   M5 Maintenance mode ──▶ M6 Specialty expansion ──▶ M7 Paper 2
                                     (add one guideline)     (non-surgical)            (cross-specialty)
```

Each milestone has an exit criterion. A milestone is done only when its criterion passes.

---

## M0 — Foundations
Four lanes that can run in parallel.

| Lane | Task | Output |
|------|------|--------|
| A | T1 store | SQLite schema, repository API, migrations, snapshot state machine |
| B | T2 harmonization | raw → {strength, direction, certainty, category}; table tests from real ACP/ESICM phrasing |
| C | T3 legacy extractor changes | `category` in prompt/parser, page/chunk tagging, optional seed, few-shot DOI exclusion + seed |
| A | T6 corpus | manifest schema + validation, `topic_id` supersession, ICU multi-xlsx loader fix |

Regression tests for T3 are mandatory (parser, extractor columns, llm_client options, ICU loader).

**Exit:** the full test suite passes. The legacy benchmark on 15 guidelines gives the same F1 (±0.01) with the new options off.

## M1 — Pipeline end to end
- T4 `ExtractorConfig`:
  - modes SINGLE_PASS / SELF_CONSISTENCY / TABLE_FIRST;
  - hash covers the config, strategy and examples, model digest, classifier SHA-256 and embedding revision.
- T7 pipeline + CLI:
  - `add <pdf>` (duplicate-safe);
  - `snapshot` (`--resume`, `--exclude`, manifest re-scan);
  - `validate` (stored runs, DOI-join hard check, paired cluster bootstrap);
  - `status` (ETA, slow and low-recall outliers, coverage);
  - `publish` (gate).
- T11 `scripts/fetch_pdfs.py`: built and tested against mocked HTTP. Not run yet.

**Exit:** with a fake strategy, `add` → `snapshot` → `validate` → `publish` runs end to end in an E2E test (no GPU). `status` shows a resumed snapshot correctly.

## M2 — Choose ExtractorVersion v1
- T5 table-first strategy (Docling/Marker), with fixture tests on ESICM and ACP PDFs.
- T9 FT v3 leave-one-guideline-out re-evaluation (GPU, 1–2 nights).
- T10: a candidate snapshot over the 12 labelled guidelines (9 ACP + 3 ESICM), qwen3 thinking off. Candidates:

  | Candidate | Note |
  |-----------|------|
  | v0 legacy few-shot + norm | baseline |
  | SC adaptive + ML filter | best F1 so far, in-sample |
  | table-first | new |
  | FT v3 | only if its LOGO result is competitive |

  Record F1, grade/certainty accuracy, ungraded P/R and hours per guideline.

**Exit:**
- v1 chosen.
- Its gate passes against v0.
- Projected full-corpus snapshot ≤ ~3 nights.
- Decision logged in the design doc.

## M3 — MVP snapshot (ESICM + SCCM + ACP, 2016–2026)
1. Build the manifest: every ESICM, SCCM and ACP guideline 2016–2026, with `topic_id` and `supersedes`.
2. Get the PDFs:
   - ESICM/ACP: open access, fetch directly.
   - SCCM: run T11 inside the VPN, then copy the PDFs to the PDF dir.
   - Resolve every "paywalled / not found" line in the fetch report.
3. Run `snapshot` with v1 → `validate` → `publish`.
4. T8 analytics + publishing: headline tables, trends by year and society, captions with counts, SCCM flagged "accuracy unvalidated", static dashboard served on this machine.
   - Run `/plan-design-review` when T8 starts.
   - Decide local-only vs LAN access.

**Exit:**
- Snapshot published with zero manifest guidelines missing an active run.
- Every figure is regenerated from the DB by `publish`.
- Dashboard reachable.

## M4 — Paper 1 (ICU + ACP)
- Fanaroff-style tables:
  - % by certainty;
  - % strong;
  - against-recommendations;
  - ungraded share;
  - trends by year;
  - ESICM vs SCCM vs ACP.
- Methods:
  - corpus and inclusion rule;
  - harmonization table;
  - denominator rule;
  - extractor version and hash;
  - validation on 12 labelled guidelines with cluster-bootstrap CIs;
  - SCCM limitation;
  - reproducibility: exact recompute from stored runs.
- Supplement: manifest, exclusions with reasons, raw-to-harmonized mapping, code SHA.
- Optional, after the snapshot: discordance analytics (TODOS.md).

**Exit:** a draft where every number comes from `publish` output of one published snapshot.

## M5 — Maintenance mode
- Routine add:
  ```
  fetch_pdfs.py (VPN if paywalled) → manifest row → evident add <pdf> → evident publish
  ```
  `add` extracts only the new guideline with the published version. `publish` recomputes stats and the dashboard.
- New edition: set `supersedes` in the manifest. The current map switches to the latest edition; trends keep both.
- Extractor upgrade: new ExtractorVersion → `snapshot` (full re-extraction) → `validate` gate → `publish`. Never mix versions.
- Runbook: add `docs/runbook.md` with these three procedures and the `status` checks.

**Exit:** a guideline published after the MVP snapshot is added in one command sequence. The dashboard updates, and `status` is clean.

## M6 — Expansion to non-surgical specialties
Order by label availability and scheme difficulty:

| Step | Specialty / societies | Scheme work | Validation |
|------|-----------------------|-------------|------------|
| 6a | Pneumology (ERS, ATS) | ABCD_123, ESC_ERS, GRADE (already detected) | 3 ERS labelled |
| 6b | Internal medicine (ACP already in; others) | GRADE | ACP labels |
| 6c | Cardiology (ESC, ACC/AHA) | COR I–III × LOE A/B/C → harmonized scale | new labels needed |
| 6d | Anesthesia (ESAIC, ASA) | mixed | new labels needed |
| 6e | Hematology (ASH, BSH) | GRADE | new labels needed |
| 6f | Remaining non-surgical | per society | per society |

Per specialty:
1. Add a scheme mapping in harmonization, with a table test.
2. Add manifest rows.
3. Fetch PDFs.
4. Opaque-table PDFs: use auto-vision or table-first (known case: NI9RV3E7).
5. Run a snapshot, then validate on whatever labels exist.
6. Mark specialties without labels "accuracy unvalidated".

Revisit the validation decision at 6a, as logged: the ceiling is ICU accuracy resting on 17 recs.

**Exit (per specialty):** mapping tested, snapshot published, and validation reported or flagged.

## M7 — Paper 2 (cross-specialty)
- Evidence quality across non-surgical medicine; comparison with Fanaroff 2008–2018 cardiology.
- Discordance map: strong recommendations on low certainty, by specialty.
- The living dashboard is cited as the maintained resource.

---

## Risks
| Risk | Effect | Mitigation |
|------|--------|------------|
| Thin ICU validation (17 recs) | weak accuracy claim | ACP labels; flag SCCM; revisit at M6 |
| Snapshot runtime > 3 nights | slow upgrades | thinking off; table-first; timing in `status` |
| Paywalled PDFs | corpus gaps | T11 + VPN; fetch report lists gaps; exclusions reported |
| Layout variance across a decade | missed recommendations | low-recall outlier flag; table-first + LLM fallback |
| Model re-pull changes weights | silent drift | digest check (`ModelDigestMismatch`) |
| Harmonization disputes (ACP insufficient, best practice) | reviewer pushback | raw grade stored; mapping table published |

## Open questions
- ACP coding of "insufficient evidence" and best-practice statements (blocks T2's final mapping).
- Dashboard access: local only or LAN (decide at T8).
- Supersession matching beyond a manual `topic_id` (deferred).
