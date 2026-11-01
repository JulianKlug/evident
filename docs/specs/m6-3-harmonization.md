# Spec: M6.3 — Harmonization of non-GRADE grading systems

**Status:** Not started
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
| Level B (B-R, B-NR) | — | — | moderate |
| Level C (C-LD) | — | — | low |
| Level C (C-EO, expert opinion) | — | — | very_low |

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

## 9. Open decisions

- ESC IIa/IIb: both weak, or a third strength level (would change `Strength` and every analysis).
- S3 single-axis grades: infer certainty from the grade's definition, or leave certainty `unmapped`.
- S5 documents: exclude, or keep as `no_recommendation`.
