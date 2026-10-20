# Paper 1 freeze (2026-10-13)

Spec: `docs/specs/m4-paper1.md` §3. Git tag `paper1-freeze` marks the commit holding this file, the manifest and the candidates.

## Source

| Item | Value |
|------|-------|
| Frozen DB | `<data root>/frozen/paper1_snapshot8_2026-10-13.sqlite` (read-only, journal mode DELETE) |
| DB SHA-256 | `145ae06aab900753313fa9df79c91aa28bc4a32d836506b0ce78c2dd3b573bc0` |
| Snapshot | 8, published |
| ExtractorVersion | v2 `39ac17093b65ea3c44a2c08b129a9882590af774e26d0e88af9411d2569f96c0` (`configs/v2_qwen38-64k_alnum.json`) |
| Model | `qwen38-64k:latest@12af24e3ee72` |
| `corpus/manifest.csv` SHA-256 | `24f955c44e4c9abbc2d06cc5dc6c08fbca70d48848c584e3eb9ad28caa1292a7` |
| `corpus/candidates.csv` SHA-256 | `323fc2ae01059ec538449f7580327637fa86b72fadb20972d426484a236f9a28` |
| Search cutoff | 2026-10-04 (`scripts/search_guidelines.py --until 2026-10-04`) |
| Held-out SCCM labels (2026-10-19) | `<data root>/intensive_care_medicine_heldout/`; SHA-256 `…002533.xlsx` `113466fb6392049784829b81cd2f26a4136f00fe8f6a5b998bc9ae8011d0dc7f`, `…006551.xlsx` `7fee54aa6c5fa5382779eceef928335fe4b10803c667c594fc70fdbba2fdd58f`, `…003914.xlsx` `fdb835f7ca6ebaeb4711ab45171aaea7e6c2511639f3bfcb98082c5cdc2051eb` |
| Package (2026-10-20) | analysis code `17efa60`; `out/paper1/analysis_manifest.json` SHA-256 `56eb7263641e4f6ad07fd64e82ae264bdfbfc7197c54afd04e6529c31a44ec07`; two runs byte-identical, DB hash unchanged; tables and validation identical to the `3738471` package (only the code SHA differs) |

Made with `sqlite3 out/evident.sqlite ".backup …"` from the live DB, then `PRAGMA journal_mode=DELETE`, `chmod 444`.

## Audit (on a scratch copy; the frozen file's hash was unchanged afterwards)

- `status`: 85 members, 85 succeeded runs, 0 failed/empty/interrupted, 0 pending, 0 excluded; no "PDF changed since its run" warning (every run holds its PDF SHA-256).
- One ExtractorVersion and one model for the snapshot.
- Editions: 75 current, 10 superseded.
- `validate --snapshot 8`: graded F1 0.939 [0.867, 0.989], P 0.885, R 1.000, combined grade accuracy 0.935, identical to the live DB.
- `publish` from the copy: tables and figures byte-identical to `out/`; the dashboard differs only in the code SHA it prints.
- `tests/evident/test_corpus.py`: manifest valid, every included candidate has a manifest row and vice versa.

## Corpus flow

| | n |
|---|---|
| Candidates | 999 (pubmed 426, pubmed_broad 360, pubmed_society 206, society_site 5, series 2) |
| Included | 85 (SCCM 39, ESICM 26, ACP 20) |
| Excluded: not_guideline | 560 |
| other_society | 148 |
| derivative | 120 |
| not_graded | 55 |
| duplicate_publication | 19 |
| endorsement | 9 |
| out_of_range | 3 |
| pdf_unavailable | 0 |

Guidelines found after the freeze (e.g. ACP menopause, 2026-10-06) belong to a later living-map update, not Paper 1.
