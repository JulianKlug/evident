# Paper 1 analysis package

Spec: `docs/specs/m4-paper1.md`. Source: `docs/paper1/freeze.md`. Every number in the draft comes from this package.

## Reproduce

From the repository root (relative paths in `configs/paper1.json`), no LLM or GPU needed:

```
python -m evident --db <data root>/frozen/paper1_snapshot8_2026-10-13.sqlite \
  publish --snapshot 8 --package out/paper1 --analysis-config configs/paper1.json \
  --heldout-labels <data root>/intensive_care_medicine_heldout
```

- The DB opens read-only; nothing in it changes (its SHA-256 is checked before and recorded after).
- Fails closed when:
  - the code has uncommitted changes (`analysis_sha` must name it);
  - the DB, manifest, candidates or held-out label files differ from the hashes in `configs/paper1.json`, or a DB `-wal` file exists;
  - a guideline lacks an active run, its recommendations or its PDF hash, or the edition count differs;
  - the held-out labels are incomplete or include other DOIs;
  - `--package DIR` exists and is neither empty nor a previous package (the swap deletes it).
- Also rewrites `out/tables`, `out/figures` and `out/dashboard` from the frozen source.
- Held-out scoring loads BioLORD on CPU at the runs' embedding revision (same similarities on any machine).
- Duplicate removal recomputes the screen with that BioLORD and applies the frozen verdicts (`corpus/dedup/paper1_verdicts.csv`, from `scripts/paper1_dedup_adjudicate.py`); a screened pair without a verdict fails the run.
- A rerun gives byte-identical files; compare against `analysis_manifest.json` → `outputs`.

## Files (`out/paper1/`)

| File | Content |
|------|---------|
| `table1_society.csv` | Per society and ALL: n/N, %, guideline-cluster 95% CI for strong, against, high/moderate |
| `table2_certainty_strength.csv` | Certainty distribution; strength × certainty |
| `table_s1_guideline_mean.csv` | Guideline-weighted mean vs pooled |
| `figure_data.csv` | Every plotted point of fig1–fig4 |
| `corpus_flow.csv`, `excluded_guidelines.csv` | Search and selection |
| `raw_harmonized_mapping.csv` | Observed raw grades → harmonized, with counts |
| `validation_development.json` | 12-guideline report (in-sample), rescored after duplicate removal |
| `validation_sccm_heldout.json` | 3 held-out SCCM guidelines, not gated |
| `spotcheck_bias.csv` | M3 spot check, exploratory |
| `duplicates.csv` | Within-guideline copies removed, beside the copy kept (spec §5) |
| `methods_facts.json` | Counts and provenance for Methods |
| `captions.md`, `draft.md` | Captions; manuscript rendered from `docs/paper1/draft.md.j2` |
| `analysis_manifest.json` | Inputs, versions, checksums; hash of every other file |

## Editing the draft

Edit `docs/paper1/draft.md.j2` (private: git-ignored, with `references.md`; its hash is in `analysis_manifest.json`), never `draft.md`. Numbers are template fields named after the export file and column, e.g. `{{ table1_society.ACP.strong_pct | pct }}`. An unknown field fails the render; `tests/evident/test_draft.py` rejects numbers typed by hand.
