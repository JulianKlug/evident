# Spec: M6.3 — Harmonization of non-GRADE grading systems

**Status:** Mapping tables, harmonize dispatch and outputs implemented (2026-11-19); decisions D1–D14 signed off
**Sources of truth:** `docs/roadmap.md` (M6, risks: harmonization disputes), `evident/harmonization.py`,
`evident/domain.py`, `docs/specs/m6-0-society-registry.md` (gold set).

## 1. Goal

Map each grading system M6 meets onto the existing harmonized scale (category, strength, direction, certainty),
keeping the raw labels, so recommendations from different societies can be counted together. Every mapping is
a published table with a test.

## 2. Starting point

- `harmonize()` raises `UnsupportedGradingFamilyError` for anything but `GradingFamily.GRADE`.
- The scale: `Strength` {strong, weak}, `Direction` {for, against}, `Certainty` {high, moderate, low, very_low},
  `Category` {graded, best_practice, no_recommendation}, `AxisStatus` {mapped, ungraded, unmapped}.
  `unmapped` already covers a system without a certainty axis.
- Legacy `evaluation/grading.py` has `ABCD_123` and `ESC_ERS` schemes with aliases (OCR repairs included).
- Survey counts (graded societies, before M6.0 review):

  | Family | Societies |
  |---|---|
  | GRADE + GRADE-adapted | 151 |
  | ESC / ACC-AHA class + level | 26 |
  | Oxford CEBM | 17 |
  | IDSA / ESMO-style roman + letter | 10 |
  | USPSTF, AAN, SIGN | 11 |
  | AWMF, NCCN | 2 |
  | Society-specific ("other") | 63 |

## 3. Scope and non-goals

**In scope:** new `GradingFamily` values, one mapping table per family, notation parsing, tests, published
mapping document.

**Out of scope:** detection (M6.2), prompts (M6.4), changing the GRADE mapping.

## 4. Grouping the long tail into shapes

The 63 society-specific systems mostly reuse five shapes. Map shapes, then let a family pick its shape and labels:

| Shape | Axes | Examples |
|---|---|---|
| S1 strength × certainty | both | GRADE, USPSTF, SORT, ASRM, AAO-HNS/AAP, IAS-USA |
| S2 class (with direction) × evidence level | both | ESC, ACC/AHA, IDSA/ESMO, AAN, CNS/AANS |
| S3 single grade derived from evidence | strength only, certainty implied | Oxford CEBM grades A–D, SIGN, AWMF A/B/0, NICE offer/consider, HAS, NASS |
| S4 evidence level only | certainty only | ADA A/B/C/E, GINA/GOLD A–D |
| S5 not a recommendation grade | none | ACR / SNMMI appropriateness 1–9, ESMO-MCBS, ESCAT |

S5 is out of scope for the harmonized scale; such documents are excluded (`not_graded`) or kept with
`Category.no_recommendation`, decided in §7.

## 5. Mappings (proposed, each needs sign-off)

Order of work = coverage: ESC_ERS/ABCD_123 (already detected in legacy) → ACC_AHA → Oxford → IDSA_ESMO → USPSTF → SIGN → AAN → AWMF → long tail.

Example, ESC / ACC-AHA:

| Raw | Strength | Direction | Certainty |
|---|---|---|---|
| Class I | strong | for | — |
| Class IIa | weak | for | — |
| Class IIb | weak | for | — |
| Class III (no benefit / harm) | strong | against | — |
| Level A | — | — | high |
| Level B-R | — | — | moderate |
| Level B-NR | — | — | low (alt. moderate, D2) |
| Level C-LD | — | — | very_low (alt. low) |
| Level C-EO (expert opinion) | — | — | very_low (D13) |

Each table states its ambiguous calls (here: IIa vs IIb collapse; C-EO). Ambiguity is resolved once,
documented, and reported in the paper (roadmap risk "harmonization disputes").

## 6. Implementation

- `GradingFamily` gains one value per family; `corpus._parse_family` accepts them.
- `harmonize()` dispatches on family to a per-family table module; raw labels stay stored unchanged.
- Notation parsing per family (e.g. `I-A`, `IIa/B-R`, `1B`, `[AII]`), with the legacy OCR aliases.
- Census regexes for M6.2 live next to each family's table.
- `docs/harmonization_tables.md`: every mapping, generated from the tables, so the published version cannot drift.

## 7. Tests

- Table test per family, from real phrasing in fetched guidelines (as M0 did for ACP/ESICM).
- Round-trip: every label in the gold set's `strength_levels` / `evidence_levels` maps or is explicitly `unmapped`.
- Golden replay for GRADE unchanged.

## 8. Exit

- Families covering ≥ 73% of graded gold-set societies mapped and tested (S1–S2 plus Oxford and IDSA).
- Generated mapping document committed.
- Remaining families listed with planned shape, or marked unsupported (their guidelines excluded with reason).

## 9. Decisions (resolved 2026-11-19)

Source of truth: `evident/grading_data/decisions.csv` (D1–D14), rendered in `docs/harmonization_tables.md`;
methods text: `docs/methods_grading_harmonisation.md`.

- ESC IIa/IIb: both weak; only the top tier is strong (D1). The alternative mapping (IIa → strong) is the
  sensitivity analysis, so `Strength` keeps two levels.
- S3 single-axis grades: letter grades map to strength (A → strong, B–D → weak; D1, D6). Certainty comes from
  the evidence level printed beside the grade. Only EPA prints the letter alone, so its certainty is read off the
  letter's definition (`scales.csv` `codes_from = strength_grade`). An Oxford/SIGN letter without a printed level
  leaves certainty `unmapped` (D9): the letter folds several levels together (SIGN B = 2++ or extrapolated 1++/1+).
- S5 systems: strength `unmapped` (D7), evidence kept where it is a real certainty scale. Exception:
  CANMAT/ISBD lines of treatment are a benefit-harm judgement and map to strength.

## 10. Implementation (as built)

- Crosswalk data: `evident/grading_data/{scales,crosswalk,families,decisions}.csv`, exported from the reviewed
  workbook by `scripts/export_grading_tables.py`; `--doc` regenerates `docs/harmonization_tables.md` (a test diffs it).
- `GradingFamily` values = coarse families with default scales (`families.csv`). A guideline overrides them with
  `strength_scale` / `evidence_scale` in `corpus/manifest.csv` (D10; store migration 5), e.g. 3- vs 4-level letters (D14).
- `harmonize(..., scales, variant)`: GRADE path unchanged; other families → `grading_tables.match` (token windows,
  filler words dropped, digit-letter split "1C") → crosswalk values → GRADE category/direction rules.
  `MappingVariant.ALTERNATIVE` returns the sensitivity mapping.
- Raw labels are stored unchanged and harmonized on read, so a mapping change never needs re-extraction.
- Outputs: `recommendations.csv` / dashboard add family, scales and the alternative mapping;
  `tables/headline_sensitivity.csv` when any non-GRADE guideline is published.
- Families: `idsa` (evidence I–III; also ESCMID/ECMM, D = against) and `esmo` (I–V) are separate, so each has
  the right default evidence scale.
- Notation (§6): combined labels are split only where a token is no code itself ("I-A", "IIa-B", "2a-B-NR",
  "1C", "AII"); the other axis is read only from tokens the first match left, never from a window that is a code
  of the first axis ("Class II a" = IIa, not level A); ties between different mappings stay unmapped. Legacy
  aliases: ESC numeric classes and the NI9RV3E7 OCR forms (ILA, LIB, ILI …), arabic IDSA quality levels.
- Direction (all families, GRADE included): "not recommended", "cannot be recommended", "do not offer",
  "not indicated", "avoid …", "should be avoided", German "soll/sollte … nicht" → against; "cannot be
  recommended" with an evidence-gap clause ("insufficient evidence") → no recommendation.
- Legacy ERS: NI9RV3E7 (ESC_ERS, 217 labelled rows) and BDYDTUHA (ABCD_123 = ESCMID SoR A–D + QoE I–III,
  60 rows) map every row through `esc_class` and `idsa` (tests need the local data dir).
- Coverage (§8): `corpus/grading/society_grading_survey.csv` (exported with the tables) → 277 of 284 graded
  surveyed societies (97.5%) have both scales mapped; the 7 others are D7 systems. Counted against the survey,
  one representative guideline per society; the exit against the M6.0 gold set stays open until it exists.
- Provenance: `grading_tables_sha256` (all crosswalk CSVs) is a release input; a table change makes `status`
  report the release stale.
- Schema v5: read-only opening needs the exact schema, so the frozen v4 paper1 DB is reproduced with the code
  at tag `paper1-freeze` (`docs/paper1/README.md`), never migrated.
- Not done here: census regexes (M6.2), `consensus_level` extraction (M6.4), per-decision sensitivity analyses
  (e.g. D1 or D2 alone; M7).
