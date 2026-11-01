# Spec: M6.2 — Grading-system detection per guideline

**Status:** Not started
**Sources of truth:** `docs/roadmap.md` (M6), `docs/specs/m6-0-society-registry.md` (gold set),
`docs/specs/m6-3-harmonization.md` (family list), `evident/corpus.py` (`grading_family` column).

## 1. Goal

For every included guideline PDF, propose the `grading_family` for its manifest row, with evidence a human can
check in seconds. Today a human types the family; at M6 scale (thousands of guidelines) that does not hold.

## 2. Starting point

- `corpus/manifest.csv` requires `grading_family` per guideline; `GradingFamily` has only `grade`.
- Societies are not consistent across their own documents (ESC guidelines graded, ESC consensus statements not;
  also FIGO, ASBMR, ESH, EFLM). Detection must be **per document**, not per society.
- Survey prototype (`.../society_survey/scripts/extract_grading.py`): qwen3:14b reads the most grading-dense
  paragraphs (≤12k chars) and returns family + levels + quote; an independent regex labels the family too.
  Observed errors:
  - graded papers labelled consensus only (ASCCP, ABC-Consensus);
  - off-list family names (`NICE`, `ESCAT`) and unnamed "other" systems;
  - methods text describing a cited system rather than the one used.

## 3. Scope and non-goals

**In scope:** detector module, CLI to fill suggestions into the manifest, evaluation against the gold set.

**Out of scope:** mapping to the harmonized scale (M6.3); changing recommendation extraction (M6.4).

## 4. Design

```
PDF ──▶ text (existing pdf loader, no network fallbacks)
         │
         ├──▶ label census: regex counts of grade labels next to recommendations
         │      "Class I/IIa/IIb/III", "LOE A/B/C", "1A..2D", "Grade A..D", "[I, A]", "soll/sollte/kann", ...
         │
         ├──▶ methods reader: LLM on the grading-dense passages + the grading table
         │      → family (closed list), levels, notation, verbatim quote, page
         │
         ▼
      combiner ──▶ suggestion + confidence ──▶ manifest `suggested_grading_family` (human confirms)
```

- **Label census** is the stronger signal: what a guideline prints next to its recommendations is what M6.4 has
  to parse. Census patterns live with each family's definition in M6.3, so adding a family adds its pattern.
- **Methods reader** runs with a closed family list (the `GradingFamily` values plus `consensus_only`), thinking off,
  temperature 0, JSON output. The quote must be found verbatim in the text; otherwise the field is dropped.
- **Combiner:**
  - agree → `high`;
  - census finds labels, reader says consensus only → census family, `low`;
  - disagree otherwise → reader family, `low`;
  - neither finds anything → `consensus_only`, `medium`.
- Layering: the detector calls the existing LLM client and PDF loader; the CLI calls the detector; nothing below
  reaches into the manifest file directly except through `corpus.py`.

### 4.1 CLI

`evident detect-grading [--manifest corpus/manifest.csv] [--dois ...]` writes `suggested_grading_family`,
`grading_confidence`, `grading_quote`, `grading_page` columns. It never overwrites `grading_family`.
`load_manifest` rejects a row whose `grading_family` is empty, so a human must confirm before extraction.

## 5. Evaluation

- Test set: one guideline per gold-set society (M6.0) with a PDF on disk, plus 30 hand-picked
  within-society contrasts (graded guideline vs ungraded statement from the same society).
- Metrics: family accuracy; accuracy on `high`-confidence suggestions; recall of `consensus_only`.
- Unit tests: census patterns per family on recorded snippets; combiner truth table; verbatim-quote check.

## 6. Exit

- Family accuracy ≥ 0.90 on the test set, and ≥ 0.97 on `high`-confidence suggestions (thresholds to confirm).
- Detector output for every included M6.5 guideline, confirmed by a human into `grading_family`.

## 7. Open decisions

- Thresholds in §6.
- Model: qwen3:14b (survey) or a larger local model; runtime budget per PDF.
