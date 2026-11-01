# Spec: M6.0 — Society registry and grading-system gold set

**Status:** Not started
**Sources of truth:** `docs/roadmap.md` (M6), survey workbook
`/mnt/data1/klug/datasets/evidence_extraction/society_grading_systems_summary.xlsx`, society list
`/mnt/data1/klug/datasets/evidence_extraction/societies_by_discipline.xlsx`.

## 1. Goal

Turn the October 2026 society survey into two versioned inputs for the rest of M6:

1. a **society registry**: which bodies M6 covers, with names, acronyms and name variants;
2. a **gold set**: the grading system each society uses, verified by a human.

The registry drives discovery (M6.1). The gold set tests grading detection (M6.2) and orders the mapping work (M6.3).

## 2. Starting point

- The society list exists only as a workbook. Its Legend says it is generated from `societies.csv`, but no
  such file exists on disk or in the repo. The `Detail` sheet (389 unique societies, CPIC and ERAS cross-listed)
  is the de facto source.
- The survey (2026-10-27) assigned one guideline and one grading family per society:

  | Outcome | Societies |
  |---|---|
  | Explicit grading identified | 280 |
  | Consensus only / not stated | 85 |
  | Not found | 24 |
  | Marked `needs_review` | 112 |
  | Confidence high / medium / low | 74 / 209 / 105 |

- Two sources fed it: Europe PMC + qwen3:14b (141 rows) and Claude web research (247 rows). Web-research quotes
  are sometimes paraphrased; some low-confidence rows were filled without opening the guideline.
- 12 of 87 societies with a prior `grading_system` value disagree with the survey.
- Scripts and raw results: `.../society_survey/scripts/`, `.../society_survey/web_research/`.

## 3. Scope and non-goals

**In scope:** registry file, review of flagged rows, gold-set file, loader + validation, per-society grading notes.

**Out of scope:** fetching more guidelines (M6.1), automatic detection (M6.2), harmonization (M6.3).

## 4. Deliverables

### 4.1 `corpus/societies.csv`

One row per society, generated once from the `Detail` sheet, then maintained in git (the workbook becomes a
derived view, not the source).

| Column | Notes |
|---|---|
| `society` | Unique code, e.g. `ESC`, `ASA-Stroke` |
| `name`, `name_variants` | `|`-separated variants, used by discovery |
| `discipline`, `subdiscipline`, `region` | As in the workbook |
| `guidelines_url` | Society's guideline index |
| `in_scope` | `yes` / `no` |
| `scope_reason` | Required when `in_scope=no`, see §4.3 |

Cross-listed societies keep one row; `discipline` takes the first listing, the second goes to a
`also_in` column.

### 4.2 `corpus/society_grading.csv` (gold set)

One row per in-scope society.

| Column | Notes |
|---|---|
| `society` | FK to `societies.csv` |
| `family` | Closed list, same values as M6.3's `GradingFamily` plus `consensus_only`, `mixed` |
| `system_name` | As the society names it |
| `strength_levels`, `evidence_levels` | `;`-separated, verbatim labels |
| `notation_example` | e.g. `Class I, Level A` |
| `reference_doi` or `reference_url` | Guideline or methods manual the label was verified on |
| `verified_by`, `verified_on` | Reviewer initials, ISO date |
| `notes` | Variation inside the society, e.g. "guidelines GRADE, consensus statements ungraded" |

`mixed` is for societies whose document types use different systems (ESC: guidelines class/level,
clinical consensus statements ungraded; FIGO, ASBMR, ESH, EFLM). Detection runs per document anyway (M6.2);
`mixed` tells the reviewer to expect it.

### 4.3 Review rules

Review every row with `needs_review = TRUE`, every disagreement with the prior column, and every
`confidence = low`. A row is verified only when the reviewer opened the reference and saw the grading
definitions (methods section, table, or methodology manual).

Set `in_scope = no` with a reason for:

| Reason | Examples from the survey |
|---|---|
| `no_guidelines` | GIN, AGREE (methods bodies), FMH-GL (portal) |
| `endorses_only` | Societies that only endorse other bodies' guidelines (e.g. WSS → AASM) |
| `not_clinical` | Research or standards bodies without clinical recommendations |

### 4.4 Loader

`evident/corpus.py` gains `load_societies()` and `load_society_grading()` with the same error style as
`load_manifest` (`ManifestError` with line-numbered problems): unique codes, FK integrity, closed family
values, required reference for verified rows.

## 5. Tests

- Loader tests: duplicate code, unknown FK, unknown family, missing reason for out-of-scope.
- Parity test: every `society` value in `corpus/manifest.csv` exists in `societies.csv`.

## 6. Exit

- `societies.csv` and `society_grading.csv` in git, loaders tested.
- Every in-scope society has a verified gold row, or a row explicitly marked `unverified` with the reason
  (paywalled methods, no guideline located). Target: at most 10% unverified.
- Family counts from the gold set replace the survey's in `docs/roadmap.md` M6.3 ordering.

## 7. Open decisions

- Who reviews (and whether a second reviewer samples 10% for agreement).
- Whether `consensus_only` societies stay in the registry for M6.1 discovery (needed if ungraded extraction
  returns in a later version) or are marked out of scope now.
