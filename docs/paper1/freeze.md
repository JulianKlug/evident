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

## Deduplicated package (2026-10-24)

`out/paper1_v2`, code `62bc0aa`, `analysis_manifest.json` SHA-256 `54e9919fbf8ae54ac85c5cb29e859a882d833c8a78eada45d77bc93ec45dd721`. Same frozen DB; adds within-guideline duplicate removal (spec §5) with verdicts `corpus/dedup/paper1_verdicts.csv` (SHA-256 `533bff8a…`). `out/paper1` (2026-10-20) is kept for comparison. Rebuilt with code `86b665e` (`--draft-template docs/paper1/draft.md.j2`): manifest SHA-256 `ba1a99061dc8777900bec6a58b148f7714a281e848bc85fe57610a0e9adc40b3`, two runs byte-identical; tables unchanged, captions and dashboard now show development F1 0.958 (after duplicate removal) instead of the stored 0.939. Three-judge verdicts (spec §5 item 5), code `c782733`: manifest SHA-256 `2285fcdc449a664f2187b450e11fde9523b8066c3ceef04124e29ae0772e2caf`; 196 statements removed (143 graded current), 52 grade-conflict groups; ALL strong 26.2%, against 18.3%, high/moderate 27.3% (without duplicate removal, Table S2: 26.1%, 17.7%, 27.5%); development F1 0.968. Superseded by spec §5 item 6 (two judges must agree, equal grades), code `f2db70c`: manifest SHA-256 `c6ccd95546222dfa89d69934aa5a9e7fa29a8b0f8ad7610fee787ba2fb4d79c6`; 140 statements removed (103 graded current), 37 duplicate pairs kept for different grades, 116 split pairs kept; ALL strong 26.3%, against 18.1%, high/moderate 27.2%; development F1 0.958.

| Current editions | Before | After |
|---|---|---|
| Graded, ALL | 1294 | 1164 |
| Strong, ALL | 26.1% | 26.4% |
| Against, ALL | 17.7% | 18.1% |
| High/moderate, ALL | 27.5% | 27.5% |
| High/moderate, ACP | 46.9% | 43.3% |
| Held-out F1 | 0.951 | 0.967 |
| Development F1 | 0.939 | 0.958 |

Removed 177 statements (ACP 4, ESICM 52, SCCM 121; 130 graded in current editions) from 178 duplicate verdicts of 1256 screened pairs; 43 groups carried different grades.
