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
- Fails closed when the DB, manifest or candidates differ from `configs/paper1.json`, a guideline lacks an active run or its PDF hash, the edition count differs, or the held-out labels are incomplete or include other DOIs.
- Also rewrites `out/tables`, `out/figures` and `out/dashboard` from the frozen source.
- Held-out scoring loads BioLORD on CPU (same similarities on any machine).
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
| `validation_development.json` | Stored 12-guideline gate report (in-sample) |
| `validation_sccm_heldout.json` | 3 held-out SCCM guidelines, not gated |
| `spotcheck_bias.csv` | M3 spot check, exploratory |
| `methods_facts.json` | Counts and provenance for Methods |
| `captions.md`, `draft.md` | Captions; manuscript rendered from `docs/paper1/draft.md.j2` |
| `analysis_manifest.json` | Inputs, versions, checksums; hash of every other file |

## Editing the draft

Edit `docs/paper1/draft.md.j2`, never `draft.md`. Numbers are template fields named after the export file and column, e.g. `{{ table1_society.ACP.strong_pct | pct }}`. An unknown field fails the render; `tests/evident/test_draft.py` rejects numbers typed by hand.
