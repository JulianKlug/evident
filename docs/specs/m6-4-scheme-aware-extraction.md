# Spec: M6.4 — Scheme-aware recommendation extraction

**Status:** Not started
**Sources of truth:** `docs/roadmap.md` (M6), `docs/specs/m6-3-harmonization.md`, `evident/extraction.py`,
`extraction/` (legacy prompts, parser, few-shot pool), `docs/progress.md` (2026-03-08 cross-scheme fallback).


> **M6.3 change (2026-11-19):** grading is resolved per guideline as a family *and* a scale pair (`strength_scale`, `evidence_scale` in `corpus/manifest.csv`, keys of `evident/grading_data/scales.csv`; empty = family default). Extraction profiles must follow the scale pair, not only the family, and raw labels must reach the store verbatim: the current prompt (`extraction/prompts.py`) coerces them to GRADE vocabulary, so the M6.3 mappings cannot apply until this changes.

## 1. Goal

Extract recommendations and their raw grades from guidelines graded in any M6.3 family, at least as well as
the GRADE extractor does on GRADE guidelines.

## 2. Starting point

- `evident/extraction.py` always builds prompts with `GRADE` few-shot examples
  (`get_few_shot_examples(GRADE, ...)`, line ~272) and the GRADE parser vocabulary.
- Legacy code has scheme-specific valid values for ABCD_123 and ESC_ERS and a cross-scheme few-shot fallback
  that adapts examples between schemes (+0.06 F1 on BDYDTUHA). NI9RV3E7 (ESC_ERS, opaque tables) needed
  auto-vision (F1 0.00 → 0.65).
- Labelled data: ACP + ICU (GRADE), 3 ERS (GRADE, ABCD_123, ESC_ERS). No labels for ACC/AHA, Oxford, IDSA, others.
- The fine-tuned `qwen3-14b-ft` was trained on GRADE-heavy data; its behaviour on other schemes is unknown.

## 3. Scope and non-goals

**In scope:** family-specific prompt text, few-shot pools, parser vocabulary, table-first / vision routing for
known table layouts, labels for the two largest new families, ExtractorVersion changes.

**Out of scope:** new model training; ungraded extraction.

## 4. Design

- The manifest row's confirmed `grading_family` (M6.2) selects a `SchemeProfile`:
  - prompt fragment describing the labels and where they appear (inline, margin column, table);
  - few-shot pool for that family; cross-scheme fallback when the pool is empty;
  - parser vocabulary + notation parser from M6.3.
- Few-shot pools must exclude the target DOI (existing rule) and be seeded (existing rule).
- ESC/ACC-AHA guidelines print recommendations in coloured tables: route them to table-first, with auto-vision
  only when text extraction loses the class column (detect by census, M6.2).
- Every profile is part of the ExtractorVersion digest, so adding or changing one forces a new snapshot (M5 §4.5).

## 5. Labels

- Label at least 3 guidelines each for the two largest new families after GRADE (expected: ESC/ACC-AHA, Oxford),
  using the existing label format and `validation.py`.
- Each specialty in M6.5 without labels stays "accuracy unvalidated" (roadmap rule).

## 6. Tests

- Prompt-building tests per profile (no LLM): correct vocabulary, few-shot exclusion, seed stability.
- Parser tests on recorded model outputs per family.
- GRADE golden replay unchanged when profile = GRADE.

## 7. Exit

- On labelled non-GRADE guidelines: recommendation F1 not below the current GRADE baseline on GRADE guidelines
  minus 0.05, and grade accuracy reported per axis.
- New ExtractorVersion validated with the M5 regression gate on the existing corpus (no GRADE regression).

## 8. Open decisions

- Whether `qwen3-14b-ft` or base qwen3:14b + few-shot is the extractor for non-GRADE families (benchmark both).
- Minimum labels per family before a specialty can drop "accuracy unvalidated".
