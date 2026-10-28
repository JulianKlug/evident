# Runbook: maintaining the living evidence map (M5)

Three procedures: **A** add a guideline, **B** add a new edition, **C** upgrade the extractor. All commands run from the repo root on the GPU machine. Tests replay A–C on a scratch DB: `tests/evident/test_runbook.py`.

```
search ─▶ curate (candidates + manifest) ─▶ PDF ─▶ add ─┬─ exit 0: published ───────────────────────────┐
                                                        └─ exit 3: judge pairs ─▶ consensus ─▶ publish ─┤
                                                                                                        ▼
                                                          status --published --check (exit 0) ─▶ progress.md
```

## Prerequisites

- `uv sync` done; `uv run pytest tests -m "not slow"` passes.
- Ollama running with the published extractor model at its recorded digest (`qwen38-64k:latest`, `12af24e3ee72`) and the second judge (`gemma3:27b`, `a418f5838eaf`): `ollama list`.
- BioLORD in the local Hugging Face cache (never downloaded at run time).
- PDF dir `/mnt/data1/klug/datasets/evidence_extraction/pdfs`; the institutional VPN for paywalled journals.
- Live DB `out/evident.sqlite`. Never point any command at the frozen M4 DB, `corpus/dedup/paper1_verdicts.csv` or `configs/paper1.json`.

**Backup first, every time.** A live WAL-mode DB must not be copied with `cp`:

```
mkdir -p artifacts/backups
sqlite3 out/evident.sqlite ".backup artifacts/backups/evident-$(date +%F-%H%M).sqlite"
```

**Exit codes** of `python -m evident`: 0 ok · 1 error or failed gate (message says why; nothing written) · 2 usage · 3 needs review (duplicate pairs without a verdict; public outputs unchanged).

## A. Add a guideline

1. **Search.** `uv run python scripts/search_guidelines.py --until YYYY-MM-DD` merges new hits into `corpus/candidates.csv` (manual decisions kept) and writes the window to `corpus/search.json`. The cutoff is the last date searched, not the publication date.
2. **Screen.** For each new row set `decision` and, for `exclude`, a `reason` (`not_guideline`, `not_graded`, `derivative`, …). Check the society's guideline page for co-publications and corrections, the first-online date, GRADE grading, full guideline vs. derivative. Write delay or exclusion reasons; never infer `supersedes_doi` from a similar title.
3. **Curate.** Add one `corpus/manifest.csv` row per included guideline: normalized DOI, `society`, `year`, `title`, `topic_id`, `grading_family`, `pdf_filename`, optional `joint_with`, optional `supersedes_doi` (procedure B). `decision=include` in the candidates. Check:
   ```
   uv run python -c "from evident.corpus import *; e=load_manifest('corpus/manifest.csv'); check_candidates(load_candidates('corpus/candidates.csv'), e); print(len(e), 'rows,', len(current_editions(e)), 'current')"
   ```
4. **PDF.** `uv run python scripts/fetch_pdfs.py --email YOU@ORG` (VPN on). Blocked or paywalled: `uv run python scripts/import_pdfs.py missing`, download by hand, then `uv run python scripts/import_pdfs.py import ~/Downloads`. Open the file: readable, the right DOI, the full guideline. Log `blocked` / `paywalled` / `not_found` outcomes; an unresolved item is not published.
5. **Add.** `uv run python -m evident add <pdf dir>/<pdf_filename>` extracts that one PDF with the published version (≈30 s–6 min), then publishes:
   ```
   10.xxxx/yyyy: added (snapshot 8)
   snapshot 8 [qwen38-64k:latest@12af24e3ee72] re-published: 19 files in out (76 guidelines; …; N duplicates removed)
   release <id> (new)
   ```
   **Exit 3** means the new guideline restates a statement and both judges must rule first. The command prints the pairs and the exact follow-up; the public outputs are untouched:
   ```
   uv run python scripts/paper1_dedup_adjudicate.py --db out/evident.sqlite --manifest corpus/manifest.csv --snapshot 8 --ledger corpus/dedup/live_verdicts.csv --dois <doi> --model qwen38-64k:latest --out corpus/dedup/judges/live-qwen38-64k-latest.csv
   uv run python scripts/paper1_dedup_adjudicate.py … --model gemma3:27b --out corpus/dedup/judges/live-gemma3-27b.csv
   uv run python scripts/paper1_dedup_consensus.py --first corpus/dedup/judges/live-qwen38-64k-latest.csv --second corpus/dedup/judges/live-gemma3-27b.csv --out corpus/dedup/live_verdicts.csv
   uv run python -m evident publish --snapshot 8
   ```
   Duplicate only when both judges agree; a split pair stays distinct; copies with different mapped grades never merge. Read `corpus/dedup/judges/disagreements.csv`. `add --no-publish` stages the run without publishing.
6. **Themes (optional).** `uv run python scripts/assign_themes.py` assigns the new current statements in `corpus/themes/assignments.csv`, then `uv run python -m evident publish --snapshot 8`. Otherwise they show as **Not yet assigned** in the exploratory theme figure; `status --check` reports the count.
7. **Check.** `uv run python -m evident status --published --check` must end `check: clean` (exit 0). Blockers: coverage gaps, changed or missing PDFs, candidate/manifest mismatch, artefact drift, missing search window, unresolved pairs, outputs older than the active runs, inputs changed since the release. Slow runs, low-recall outliers and unassigned themes are warnings for review.
8. **Verify** `out/dashboard/index.html` (served locally: `python -m http.server -d out/dashboard`): the guideline is listed, the counts moved as expected, the provenance line shows snapshot, version, search cutoff and edition rule. Public hosting needs a separate decision (`TODOS.md`).
9. **Record.** Run the same `add` again: `already_present`, `release <id> (unchanged)`. Commit the curated CSVs, `corpus/search.json` and `corpus/dedup/live_verdicts.csv`. Add a `docs/progress.md` entry: commands, run times, DOI, before/after counts, release id from `out/release.json`. The release is archived in `artifacts/releases/<id>/` (record + DB backup).

## B. Add a new edition

A **whole replacement** (the new edition replaces every recommendation of the old one):

1. Procedure A with the new row's `topic_id` equal to the old edition's and `supersedes_doi` = the old DOI (several: `a;b`). The manifest refuses unknown DOIs, cycles, another topic, a not-earlier year, or an older edition of the topic that no row supersedes.
2. After publishing check `out/release.json` → `counts`: `current` unchanged, `superseded` + 1. In `out/tables/guidelines.csv` the old DOI is superseded, the new one current; `out/tables/trends.csv` keeps both years. The old guideline, PDF and run are kept.

A **focused update** (only some questions revised): give it its own `topic_id`, no `supersedes_doi`. Both stay current; note the possible double counting in `docs/progress.md`. Replacing single recommendations is out of scope (`TODOS.md`).

## C. Upgrade the extractor

Any change to the model, prompt, few-shot pool, extraction algorithm or a result-relevant mapping is a new ExtractorVersion. Never `add --force` for it; never mix versions in one release.

1. Backup. `uv run python -m evident snapshot --config configs/<new>.json` re-extracts every manifest edition (~1 h per 100 guidelines). After a crash or failures: `uv run python -m evident snapshot --resume` (`--exclude DOI --reason TEXT` only for a deliberate, documented exclusion).
2. `uv run python -m evident validate --snapshot <id>`: gate against the published snapshot.
3. Duplicates: new texts need new verdicts. Count first, then run both judges and the consensus as in A.5 with `--snapshot <id>` and without `--dois`:
   `uv run python scripts/paper1_dedup_adjudicate.py --db out/evident.sqlite --snapshot <id> --ledger corpus/dedup/live_verdicts.csv --out /dev/null --dry-run`
4. `uv run python -m evident publish --snapshot <id>`. It refuses (exit 1, nothing written) on a failed gate, a coverage gap, PDF or artefact drift; exit 3 on unjudged pairs. `--accept-regression REASON` only after a human review, with the reason recorded.
5. A candidate that fails: `uv run python -m evident reject --snapshot <id> --reason TEXT`. The published snapshot and its outputs stay.
6. `status --published --check` clean. In `docs/progress.md`: both version ids, gate result, run time, headline change (`out/tables/headline.csv` before vs after), release id.

## Troubleshooting

| Symptom | Action |
|---|---|
| `PDF of <doi> differs from the bytes its active run read` / blocker `PDF missing or changed` | (a) restore the recorded bytes: `rebind-pdf <backup copy>` copies them back (`restored`; check `sha256sum` = `inputs.pdfs` in `out/release.json`); else (b) `uv run python -m evident rebind-pdf <pdf>`: accepted only when every rebuilt prompt matches the run's (no model call), logged in `corpus/pdf_rebinds.csv`, then `publish --snapshot <id>` to record the new hash; else (c) `add --force <pdf>` re-extracts, then judge new pairs. |
| `rebind-pdf`: `changes the model input (chunk N prompt differs)` | The text really changed (e.g. a download stamp). Use (a) or (c). |
| `model_digest changed since the version was registered` | The model was re-pulled. Restore the recorded digest; otherwise it is an extractor upgrade (C). |
| `No manifest row with pdf_filename …` / missing or inaccessible PDF | Fix the manifest filename or import the PDF (A.4); never mark it published. |
| `failed` / `empty` run in `status` | Read the error; fix (PDF, Ollama), then `add --force <pdf>`. The public outputs stay those of the last good release. |
| exit 3 / `verdicts for the same texts` | Run both judges and the consensus (A.5). A conflicting ledger row is a hand edit: restore `live_verdicts.csv` from git. |
| `status` shows a BUILDING or COMPLETE candidate | `status --published` shows the release. Finish with `snapshot --resume`, or `reject --snapshot ID --reason TEXT`. |
| `Publication blocked` / render error | Nothing was written; the previous `out/` and `out/release.json` stand. Fix the listed cause and publish again. |
| `no release record` / `public outputs predate the active runs` / `changed since release` / `<file>: differs from release` | `uv run python -m evident publish --snapshot <published id>` (after `publish --package`, which renders the M4 outputs into `out/`, always). |
