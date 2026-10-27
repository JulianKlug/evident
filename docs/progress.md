# Extraction Pipeline Progress

## 2025-03-04: Baseline Benchmark

Ran full benchmark: 3 models x 2 strategies x 6 guidelines (3 ACP + 3 ERS).

### Best results per guideline (from `benchmark_results_all.csv`)

| Guideline | Pages | GT Recs | Best Model | Strategy | F1 | P | R | Grade | Level |
|-----------|-------|---------|------------|----------|-----|-----|-----|-------|-------|
| ACP:XLMXNL32 | 29 | 6 | qwen3:8b | few_shot | 0.71 | 0.62 | 0.83 | 1.00 | 1.00 |
| ACP:8J2P9MD8 | 11 | 4 | qwen3:8b | few_shot | 0.89 | 0.80 | 1.00 | 0.75 | 1.00 |
| ACP:8V9WED94 | 20 | 3 | deepseek-r1:32b | few_shot | 0.80 | 1.00 | 0.67 | 0.50 | 0.50 |
| ERS:NI9RV3E7 | 80 | 217 | llama3.2 | few_shot | 0.16 | 0.15 | 0.18 | 0.00 | 0.00 |
| ERS:CMCZFLU4 | 17 | 12 | deepseek-r1:32b | few_shot | 0.74 | 0.67 | 0.83 | 0.00 | 0.00 |
| ERS:BDYDTUHA | 24 | 60 | deepseek-r1:32b | zero_shot | 0.76 | 0.70 | 0.83 | 0.94 | 0.84 |

**Key issues identified:**
- NI9RV3E7 (80 pages, 217 recs) has near-zero recall across all models
- ERS grade/level accuracy near 0 even with good F1
- Evaluation takes 16+ min for large guidelines (pairwise similarity)
- Context window defaults to ~4K tokens despite models supporting 16-32K

---

## 2025-03-05: Phase 1 Implementation

### Changes made

#### 1A. Context Window Fix (`extraction/llm_client.py`)
- Added `num_ctx` parameter to `generate()` and `generate_json()`
- Only passes `num_ctx` to Ollama when explicitly set (avoids performance regression)
- Added 4 new models to `AVAILABLE_MODELS`: qwen3:14b, gemma3:27b, mistral-small3.2:24b, qwen3:30b-a3b

#### 1B. Batch Similarity (`evaluation/matching.py`, `extraction/benchmark.py`)
- Added `encode_batch()` to `_BioLORDSimilarityModel` using sentence-transformers batch encoding + L2 normalization
- Updated `build_similarity_matrix()` to use batch encoding when available, falling back to pairwise
- **Impact: NI9RV3E7 evaluation from 966s to <1s**

#### 1C. Improved Prompts (`extraction/prompts.py`)
- Precise recommendation definition: "actionable statement that directs clinical practice, explicitly graded"
- Verbatim extraction instruction
- Negative examples: "Do NOT extract background statements, evidence summaries, section headers"
- Grade/level constraints from the grading scheme

#### 2A. Multi-Page Chunking (`extraction/extractor.py`)
- Added `_chunk_pages(pages, pages_per_chunk, overlap)` for grouping pages
- `extract_guideline()` accepts `pages_per_chunk` parameter (default=1 preserves behavior)
- Only passes `num_ctx` when multi-page chunking is active

#### 2B. JSON Structured Output (`extraction/llm_client.py`, `extraction/response_parser.py`)
- Added `generate_json()` using Ollama's `format` parameter with JSON schema
- Added `parse_json_response()` for structured output parsing
- Activated via `output_format="json"` in extractor

#### 3A. Post-Extraction Normalization (`extraction/postprocessing.py`, `evaluation/grading.py`)
- New `normalize_extracted_grades()` applies scheme normalization to extracted values
- Extended GRADE aliases: "strong recommendation for/against", "conditional", certainty variants
- Extended ABCD_123 aliases: "level of evidence N", "++", "+"

#### 3B. Few-Shot Examples (`extraction/datasets.py`, `extraction/prompts.py`)
- `get_few_shot_examples()` accepts `include_source_text` parameter
- Prompt renders source text context when available

#### 4A. New Models (`extraction/llm_client.py`)
- Added to AVAILABLE_MODELS: qwen3:14b, gemma3:27b, mistral-small3.2:24b, qwen3:30b-a3b

#### 4B. Two-Pass Extraction (`extraction/two_pass.py`)
- Page classification with fast model (llama3.2) + extraction from relevant pages only
- `two_pass_extract()` function with context expansion around relevant pages

#### CLI Updates (`run_benchmark.py`)
- New flags: `--pages-per-chunk N`, `--json`, `--normalize`, `--two-pass`

### Phase 1 Benchmark: qwen3:8b few-shot + normalize

| Guideline | Baseline F1 | Phase 1 F1 | Delta | Baseline Grade | Phase 1 Grade |
|-----------|------------|-----------|-------|---------------|--------------|
| ACP:XLMXNL32 | 0.71 | 0.59 | -0.12 | 1.00 | 1.00 |
| ACP:8J2P9MD8 | 0.89 | 0.80 | -0.09 | 0.75 | 0.75 |
| ACP:8V9WED94 | 0.67 | 0.29 | -0.38 | 0.50 | 0.50 |
| ERS:NI9RV3E7 | 0.01 | 0.00 | -0.01 | 0.00 | 0.00 |
| ERS:CMCZFLU4 | 0.48 | **0.75** | +0.27 | 0.25 | 0.00 |
| ERS:BDYDTUHA | 0.62 | **0.67** | +0.05 | 1.00 | 0.83 |

### Phase 2 Benchmark: qwen3:14b few-shot + normalize

| Guideline | Baseline F1 | qwen3:14b F1 | Delta | Grade | Level |
|-----------|------------|-------------|-------|-------|-------|
| ACP:XLMXNL32 | 0.71 | **0.91** | +0.20 | 1.00 | 1.00 |
| ACP:8J2P9MD8 | 0.89 | 0.73 | -0.16 | 0.75 | 1.00 |
| ACP:8V9WED94 | 0.67 | 0.57 | -0.10 | 0.50 | 0.50 |
| ERS:NI9RV3E7 | 0.01 | 0.00 | -0.01 | 0.00 | 0.00 |
| ERS:CMCZFLU4 | 0.48 | 0.13 | -0.35 | 0.00 | 0.00 |
| ERS:BDYDTUHA | 0.62 | 0.58 | -0.04 | 0.89 | 0.89 |

---

## Key Insights (Phase 1–2)

### NI9RV3E7 is a PDF extraction problem
The 80-page ERS pulmonary hypertension guideline has 217 ground truth recommendations, all presented in **structured tables** within the PDF. The PyPDF text extraction captures surrounding text and table footnotes but not the table rows containing the actual recommendations. No LLM or prompt changes can fix this — it requires vision-based extraction.

### No single model dominates
- **qwen3:14b**: Best for ACP (high precision, F1=0.91 on XLMXNL32)
- **deepseek-r1:32b**: Best for ERS (better recall on CMCZFLU4=0.74, BDYDTUHA=0.70)
- **qwen3:8b + improved prompts**: Best for CMCZFLU4 (F1=0.75)

### Batch similarity is the biggest engineering win
Evaluation speedup from pairwise to batch: **966s to <1s** for the largest guideline.

---

## 2026-03-05: Grading Scheme Fix (CRITICAL)

### Root cause of 0% ERS grade accuracy

Discovered that ERS guidelines use **three different grading schemes**, but all were hardcoded as `ABCD_123`:

| Guideline | Was | Should Be | Impact |
|-----------|-----|-----------|--------|
| NI9RV3E7 | ABCD_123 | **ESC_ERS** (I/IIa/IIb/III + A/B/C) | Grade matching was impossible |
| CMCZFLU4 | ABCD_123 | **GRADE** (Strong/Conditional + confidence levels) | Grade matching was impossible |
| BDYDTUHA | ABCD_123 | ABCD_123 (correct) | Grade normalization incomplete |

### Changes made

#### Auto-Detect Grading Scheme (`extraction/datasets.py`)
- `_detect_grading_scheme()`: inspects GT grade values to determine GRADE vs ESC_ERS vs ABCD_123
- `_normalize_ers_gt()`: applies scheme-specific normalization per detected scheme
- OCR artifact fixes for NI9RV3E7: ILA→IIa, ILB→IIb, ILI→III, LIB→IIb, LLA→IIa
- Drops rows with invalid "0.0" values
- ACP GT normalization: truncation fix ("onditional recommendation"→"Weak For", "trong"→"Strong For")
- Trailing punctuation stripped

#### Extended Grade/Level Aliases (`evaluation/grading.py`)
- ESC_ERS: "class 1"→I, "class 2a"→IIa, "ii a"→IIa, level of evidence aliases
- GRADE: confidence-style levels ("low confidence in estimates of effect"→Low)
- GRADE: GT truncation artifacts ("onditional recommendation"→Weak For)

### Benchmark: Corrected Schemes (qwen3:14b few-shot + normalize)

| Guideline | Scheme | Baseline F1 | Fixed F1 | Baseline Grade | Fixed Grade |
|-----------|--------|-------------|----------|----------------|-------------|
| ACP:XLMXNL32 | grade | 0.71 | **0.91** | 1.00 | **1.00** |
| ACP:8J2P9MD8 | grade | 0.89 | 0.67 | 0.75 | 0.75 |
| ACP:8V9WED94 | grade | 0.67 | 0.67 | 0.50 | 0.50 |
| ERS:NI9RV3E7 | esc_ers | 0.01 | 0.00 | 0.00 | 0.00 |
| ERS:CMCZFLU4 | grade | 0.48 | **0.83** | 0.00 | **1.00** |
| ERS:BDYDTUHA | abcd_123 | 0.62 | **0.68** | 0.00 | **0.91** |

---

## 2026-03-06: Full Model Comparison (7 models, corrected schemes)

### Results: F1 Score (excluding NI9RV3E7)

| Model | Size | XLMXNL32 | 8J2P9MD8 | 8V9WED94 | CMCZFLU4 | BDYDTUHA | Avg F1 |
|-------|------|----------|----------|----------|----------|----------|--------|
| **qwen3:14b** | 9.3GB | **0.92** | **0.89** | 0.80 | **0.87** | 0.68 | **0.83** |
| qwen3:8b | 5.2GB | 0.80 | 0.62 | 0.80 | 0.83 | 0.60 | 0.73 |
| deepseek-r1:32b | 19GB | 0.86 | 0.67 | 0.27 | 0.69 | 0.70 | 0.64 |
| mistral-small3.2:24b | 15GB | **1.00** | 0.67 | 0.80 | 0.47 | 0.44 | 0.68 |
| gemma3:27b | 17GB | 0.43 | 0.80 | 0.44 | 0.49 | **0.76** | 0.58 |
| qwen3:30b-a3b | 18GB | 0.80 | 0.00 | **1.00** | **0.95** | 0.00 | 0.55 |

### Results: Grade Accuracy

| Model | XLMXNL32 | 8J2P9MD8 | 8V9WED94 | CMCZFLU4 | BDYDTUHA | Avg Grade |
|-------|----------|----------|----------|----------|----------|-----------|
| **deepseek-r1:32b** | 1.00 | 1.00 | 1.00 | 0.90 | 0.98 | **0.98** |
| qwen3:8b | 0.83 | 1.00 | 1.00 | 0.90 | 0.59 | 0.86 |
| **qwen3:14b** | 1.00 | 1.00 | 0.50 | 0.80 | 0.91 | **0.84** |
| gemma3:27b | 1.00 | 0.75 | 0.50 | 0.70 | 1.00 | 0.79 |
| mistral-small3.2:24b | 1.00 | 0.75 | 0.50 | 0.62 | 0.94 | 0.76 |
| qwen3:30b-a3b | 1.00 | 0.00 | 0.50 | 0.67 | 0.00 | 0.43 |

### Speed Comparison (total time for 6 guidelines)

| Model | Total Time |
|-------|-----------|
| mistral-small3.2:24b | ~260s (fastest) |
| gemma3:27b | ~476s |
| qwen3:14b | ~2180s |
| deepseek-r1:32b | ~2579s |
| qwen3:8b | ~5759s |
| qwen3:30b-a3b | ~7923s (slowest) |

---

## 2026-03-06: Vision-Based Table Extraction for NI9RV3E7

### Problem

NI9RV3E7 stores all 217 recommendations in **tables rendered as vector graphics**. PyPDF, Docling, pdfplumber, Tesseract, and RapidOCR all failed to extract cell content.

### Approach: Page Image → Vision LLM

1. **pdfplumber** detects which pages contain tables (35 of 80 pages)
2. Full pages rendered as PNG images at 300 DPI
3. Vision LLM reads each page image and extracts recommendations
4. Results parsed, deduplicated, and combined with text-based extraction

### Implementation
- `extraction/vision_extractor.py`: vision extraction pipeline
- `extraction/pdf_loader.py`: `load_pdf_table_images()` for page rendering
- `run_benchmark.py`: `--vision`, `--vision-model`, `--auto-vision` flags

---

## 2026-03-08: Benchmark Batch — JSON, Two-Pass, Ensemble, Vision Models

### JSON Structured Output (`--json`)

| Guideline | Baseline F1 | JSON F1 | Delta |
|-----------|------------|---------|-------|
| ACP:XLMXNL32 | **0.92** | 0.40 | -0.52 |
| ACP:8J2P9MD8 | **0.89** | 0.67 | -0.22 |
| ACP:8V9WED94 | 0.67 | 0.29 | -0.38 |
| ERS:CMCZFLU4 | **0.87** | 0.69 | -0.18 |
| ERS:BDYDTUHA | **0.68** | 0.64 | -0.04 |
| **Average** | **0.83** | **0.47** | **-0.36** |

**Verdict: CLOSED.** JSON causes massive over-extraction.

### Two-Pass Extraction (`--two-pass`)

| Guideline | Baseline F1 | Two-Pass F1 | Delta |
|-----------|------------|------------|-------|
| ACP:XLMXNL32 | **0.92** | 0.00 | -0.92 |
| ACP:8J2P9MD8 | 0.89 | 0.89 | 0.00 |
| ACP:8V9WED94 | 0.67 | 0.67 | 0.00 |
| ERS:CMCZFLU4 | 0.87 | **0.91** | +0.04 |
| ERS:BDYDTUHA | **0.68** | 0.34 | -0.34 |
| **Average** | **0.83** | **0.47** | **-0.36** |

**Verdict: CLOSED.** llama3.2 page classifier too conservative, kills recall.

### Ensemble (qwen3:14b + deepseek-r1:32b)

| Guideline | Baseline F1 | Ensemble F1 | Delta |
|-----------|------------|------------|-------|
| ACP:XLMXNL32 | **0.92** | 0.55 | -0.37 |
| ACP:8J2P9MD8 | **0.89** | 0.32 | -0.57 |
| ACP:8V9WED94 | 0.67 | 0.40 | -0.27 |
| ERS:CMCZFLU4 | **0.87** | 0.50 | -0.37 |
| ERS:BDYDTUHA | 0.68 | 0.63 | -0.05 |
| **Average** | **0.83** | **0.42** | **-0.41** |

**Verdict: CLOSED.** Union+dedup adds too many false positives from deepseek-r1:32b.

### Vision Model Comparison on NI9RV3E7

| Vision Model | Size | Extracted | F1 | P | R | Grade | Level |
|-------------|------|-----------|------|------|------|-------|-------|
| **mistral-small3.2:24b** | 15GB | 151 | **0.65** | **0.79** | 0.55 | **0.94** | **0.97** |
| gemma3:27b | 17GB | 79 | 0.30 | 0.57 | 0.21 | 0.69 | 0.84 |
| qwen2.5vl:7b | 4.7GB | CRASH | — | — | — | — | — |

**Best vision model: mistral-small3.2:24b** — 2.2x better F1, 3.8x better precision, much less hallucination.

---

## 2026-03-08: Cross-Scheme Few-Shot Fallback

### Problem

BDYDTUHA (only ABCD_123 guideline) and NI9RV3E7 (only ESC_ERS guideline) had zero same-scheme few-shot examples available → silently fell back to zero-shot.

### Solution

Added cross-scheme fallback in `get_few_shot_examples()`: when no same-scheme examples exist, adapts examples from other schemes using grade/level mapping tables.

### Results (BDYDTUHA with cross-scheme few-shot)

| Metric | Before (silent zero-shot) | After (cross-scheme few-shot) |
|--------|--------------------------|-------------------------------|
| F1 | 0.68 | **0.74** |
| Recall | 0.58 | **0.68** |
| Precision | 0.81 | 0.81 |
| Grade | 0.91 | **0.98** |

---

## 2026-03-09: Self-Consistency Voting (CLOSED)

### Approach

Run N stochastic extractions (temp=0.3) per chunk, cluster with BioLORD similarity, keep recommendations appearing in ≥ consensus threshold of runs.

### Results (3 samples, temp=0.3, consensus=2/3)

| Guideline | Baseline F1 | SC F1 | Delta |
|-----------|------------|-------|-------|
| ACP:XLMXNL32 | 0.92 | 0.86 | -0.06 |
| ACP:8J2P9MD8 | 0.89 | 0.89 | 0.00 |
| ACP:8V9WED94 | 0.80 | 0.80 | 0.00 |
| ERS:CMCZFLU4 | 0.87 | 0.87 | 0.00 |
| ERS:BDYDTUHA | **0.74** | 0.49 | -0.25 |
| **Average** | **0.83** | **0.78** | **-0.05** |

**Verdict:** Stochastic runs too similar at temp=0.3. Consensus filtering devastates BDYDTUHA recall (0.68→0.42). 3x slower.

---

## 2026-03-10: Token Overlap Verification (CLOSED)

Post-extraction verification checking each recommendation against source PDF text using token overlap ratio. Near-zero impact: only 1 rec removed across all runs. Precision +0.001, F1 unchanged. Kept as zero-cost safety net (`--verify` flag).

**Key insight:** False positives are real PDF text (background statements) misclassified as recommendations — not hallucinations. Token overlap can't distinguish these.

---

## 2026-03-11: Parser Hardening (KEPT)

Defensive improvements to `extraction/response_parser.py`: fuzzy sentinel detection, markdown stripping, numbered prefix handling, whitespace normalization. Zero-cost, no regression.

---

## 2026-03-11: Improved Few-Shot Example Selection (REVERTED)

### Experiment 1: Stratified sampling + negative prompt examples

| Guideline | Baseline F1 | Combined F1 | Delta |
|-----------|------------|-------------|-------|
| ACP:XLMXNL32 | 0.92 | 0.55 | -0.37 |
| ACP:8J2P9MD8 | 0.89 | 0.67 | -0.22 |
| ACP:8V9WED94 | 0.67 | 0.67 | 0.00 |
| ERS:CMCZFLU4 | 0.87 | 0.75 | -0.12 |
| ERS:BDYDTUHA | 0.74 | 0.49 | -0.25 |
| **Average** | **0.83** | **0.63** | **-0.20** |

### Experiment 2: Stratified sampling only

| Guideline | Baseline F1 | Stratified F1 | Delta |
|-----------|------------|---------------|-------|
| ACP:XLMXNL32 | 0.92 | 0.55 | -0.37 |
| ACP:8J2P9MD8 | 0.89 | 0.67 | -0.22 |
| ACP:8V9WED94 | 0.67 | 0.67 | 0.00 |
| ERS:CMCZFLU4 | 0.87 | 0.69 | -0.18 |
| ERS:BDYDTUHA | 0.74 | 0.69 | -0.05 |
| **Average** | **0.83** | **0.65** | **-0.18** |

**Both reverted.** The specific examples matter more than their diversity. The precision gap is a classification problem not addressable through example selection.

---

## 2026-03-12: A/B Validation on 15-Dataset Benchmark

Expanded from 5-6 guidelines to **15 datasets** (9 ACP + 3 ERS + 3 ICU, all with PDFs). Ran 7 A/B tests comparing each feature variant against the baseline on all 15 guidelines.

### Results (excluding NI9RV3E7)

| Test | Avg F1 | ΔF1 | Avg P | Avg R | Avg Grade | Prior (5-6 ds) | Validated? |
|------|--------|-----|-------|-------|-----------|-----------------|------------|
| **baseline** | **0.707** | — | 0.608 | 0.936 | 0.881 | 0.83 | — |
| self_consistency | **0.750** | +0.043 | 0.820 | 0.766 | 0.891 | 0.78 (-0.05) | **Reversed** |
| no_normalize | 0.726 | +0.019 | 0.639 | 0.936 | 0.860 | — | ✓ (trades grade for F1) |
| zero_shot | 0.697 | -0.010 | 0.600 | 0.936 | 0.784 | worse | Marginal |
| verify | 0.694 | -0.013 | 0.590 | 0.924 | 0.897 | neutral | ✓ |
| chunking_3page | 0.678 | -0.029 | 0.577 | 0.917 | 0.948 | negative | ✓ |
| auto_vision | 0.663 | -0.045 | 0.562 | 0.931 | 0.872 | neutral | ✗ (hurts text guidelines) |

### Key Findings

1. **Baseline F1 dropped from 0.83 → 0.71** when expanding from 5 to 15 guidelines. The new ACP/ICU guidelines have lower precision (many false positives). ICU guidelines especially tough: P=0.17–0.50.

2. **Self-consistency reversed from -0.05 to +0.04.** On 15 datasets, the precision boost (0.61→0.82) outweighs the recall loss (0.94→0.77). Previously tested on 5 guidelines where the best ones (XK8ZAXYM, 48AJE2AR) got hurt; on 15 datasets the many imprecise guidelines benefit more.

3. **Auto-vision hurts text-only guidelines** (Δ=-0.045 excl NI9RV3E7). Some text guidelines trigger false table detection, degrading results. NI9RV3E7 itself: F1=0.00→0.65 confirmed.

4. **Normalization trades F1 for grade accuracy**: no_normalize has +0.019 F1 but -0.021 grade. CMCZFLU4 loses 0.30 grade accuracy without normalization.

---

## Summary of All Approaches

| Approach | Avg F1 (5ds) | Avg F1 (15ds) | vs Baseline | Status |
|----------|-------------|--------------|-------------|--------|
| **SC Adaptive + ML filter** | — | **0.860 (+0.153)** | **positive** | **BEST** |
| **Adaptive self-consistency** | — | **0.783 (+0.076)** | **positive** | **KEPT** |
| **Self-consistency voting** | 0.78 | **0.750** | **+0.043** | **RE-OPENED** |
| **ML filter (BioLORD+LR)** | — | **0.732 (+0.025)** | **positive** | **KEPT** |
| **Baseline (few-shot + normalize)** | **0.83** | **0.707** | — | **BEST (simple)** |
| Context-aware oracle (deepseek-r1:32b) | — | 0.690 (-0.017) | grade +0.006 | CLOSED |
| Cross-scheme few-shot fallback | +0.06 BDYDTUHA | included | positive | **KEPT** |
| Parser hardening | 0.00 | included | neutral | **KEPT** (defensive) |
| Token overlap verification | +0.001 | -0.013 | neutral | **KEPT** (safety net) |
| Auto-vision (mistral, NI9RV3E7 only) | 0.65 NI9RV3E7 | 0.65 NI9RV3E7 | helps NI9RV3E7 | **KEPT** (vision only) |
| Stratified few-shot selection | 0.65 | — | negative | CLOSED |
| Negative prompt examples | 0.63 | — | negative | CLOSED |
| JSON structured output | 0.47 | — | negative | CLOSED |
| Two-pass extraction | 0.47 | — | negative | CLOSED |
| Ensemble (union+dedup) | 0.42 | — | negative | CLOSED |
| 3-page chunking | < baseline | 0.678 (-0.029) | negative | CLOSED |
| CoT prompt | < baseline | — | negative | CLOSED |
| JSON schema enforcement | — | 0.498 (-0.209) | negative | CLOSED |
| Post-extraction filter (qwen3:8b) | — | 0.699 (-0.008) | negative | CLOSED |
| Grading oracle (deepseek-r1:32b) | — | 0.676 (-0.031) | negative | CLOSED |

---

## 2026-03-13: Phase 1 Improvements Evaluation

Implemented and tested 3 new features on the full 15-dataset benchmark.

### 1. JSON Schema Enforcement (re-test)

Previous JSON result (F1=0.47) predated Ollama's `format=schema` enforcement. Re-tested with `additionalProperties: false` in the JSON schema.

| Metric | Baseline | JSON Schema | Delta |
|--------|----------|-------------|-------|
| Avg F1 | 0.707 | 0.498 | **-0.209** |
| Avg P | 0.608 | 0.378 | -0.230 |
| Avg R | 0.936 | 0.939 | +0.003 |
| Avg Grade | 0.881 | 0.903 | +0.022 |

**Verdict: CLOSED.** Schema enforcement doesn't fix the core problem — qwen3:14b still massively over-extracts with JSON output (P=0.38). Worse than the original JSON test.

### 2. Post-Extraction Classification Filter

Binary YES/NO classifier using qwen3:8b to filter each extracted candidate as "recommendation" or "not recommendation."

| Metric | Baseline | Post-Filter | Delta |
|--------|----------|-------------|-------|
| Avg F1 | 0.707 | 0.699 | **-0.008** |
| Avg P | 0.608 | 0.630 | +0.022 |
| Avg R | 0.936 | 0.883 | -0.053 |
| Avg Grade | 0.881 | 0.901 | +0.020 |

**Verdict: CLOSED.** qwen3:8b said YES to everything for 9/14 guidelines (0 removed). When it did filter (BDYDTUHA: removed 12/43), recall dropped aggressively. Additionally 6× slower due to qwen3:8b's thinking mode (~30-60s per classification call).

### 3. Adaptive Self-Consistency Thresholds (NEW BEST)

Replaces fixed consensus=2/3 with cluster-count-adaptive thresholds:
- ≤5 clusters: threshold=1 (prevents stochastic deletion on small datasets)
- 6–30 clusters: threshold=ceil(n_samples × 0.5) (moderate, same as default)
- >30 clusters: threshold=ceil(n_samples × 0.34) (lenient for large datasets)

| Metric | Baseline | SC Fixed | SC Adaptive | Δ vs Baseline |
|--------|----------|----------|-------------|---------------|
| Avg F1 | 0.707 | 0.750 | **0.783** | **+0.076** |
| Avg P | 0.608 | 0.820 | 0.788 | +0.180 |
| Avg R | 0.936 | 0.766 | 0.841 | -0.095 |
| Avg Grade | 0.881 | 0.891 | 0.892 | +0.011 |

**Verdict: NEW BEST CONFIG.** F1=0.783 (+0.076 over baseline, +0.033 over fixed SC). Better recall than fixed SC (0.841 vs 0.766) while maintaining most of the precision gain.

---

## 2026-03-13: Grading Oracle (CLOSED)

### Hypothesis

deepseek-r1:32b had 0.98 grade accuracy as primary extractor (but poor precision). Use it as a post-extraction re-grading step: after qwen3:14b extracts recommendations, send each to deepseek-r1:32b to verify/correct the grade and level. Expected to improve grade accuracy from 0.881→0.95+ without affecting F1.

### Implementation

New module `extraction/grading_oracle.py`:
- `regrade_recommendations(df, scheme, model)` — iterates rows, prompts oracle with recommendation text + current grade/level + valid values, parses response (stripping `<think>` tags), validates against scheme, updates only when valid.
- Fail-safe: keeps original values if parsing fails or value is invalid.
- Pipeline position: after normalization (last step).
- CLI: `--grading-oracle`, `--oracle-model`
- A/B configs: `grading_oracle` (priority=15), `sc_adaptive_oracle` (priority=16)

### Results (15-dataset benchmark, baseline + oracle)

| Metric | Baseline | Oracle | Delta |
|--------|----------|--------|-------|
| Avg F1 | 0.707 | 0.676 | **-0.031** |
| Avg P | 0.608 | 0.552 | -0.056 |
| Avg R | 0.936 | 0.949 | +0.013 |
| Avg Grade | 0.881 | 0.874 | **-0.007** |
| Avg Level | 0.940 | 0.839 | **-0.101** |

### Per-Guideline Detail

| Guideline | Base F1 | Oracle F1 | ΔF1 | Base Grade | Oracle Grade | ΔGrade |
|-----------|---------|-----------|-----|------------|--------------|--------|
| ACP:XK8ZAXYM | 1.000 | 0.750 | -0.250 | 1.000 | 1.000 | 0.000 |
| ACP:PAEHSPH3 | 0.857 | 0.667 | -0.190 | 1.000 | 1.000 | 0.000 |
| ACP:XLMXNL32 | 0.750 | 0.632 | -0.118 | 0.833 | 1.000 | +0.167 |
| ERS:BDYDTUHA | 0.699 | 0.797 | +0.098 | 0.972 | 0.809 | -0.164 |
| ACP:8J2P9MD8 | 0.667 | 0.727 | +0.061 | 1.000 | 1.000 | 0.000 |
| ERS:CMCZFLU4 | 0.690 | 0.690 | 0.000 | 0.700 | 0.600 | -0.100 |

**Verdict: CLOSED.** The oracle confidently overwrites correct values. Without source text in the prompt, deepseek-r1:32b guesses based on recommendation text alone — it changed 13 grades and 25 levels across 15 guidelines, making more wrong than right. Level accuracy dropped catastrophically (-0.101). The hypothesis that deepseek-r1:32b's high grade accuracy as extractor would transfer to a re-grading role was not validated — its accuracy came from reading the source PDF, not from domain knowledge about what grades recommendations "should" have.

**Key insight:** Grade accuracy as an extractor ≠ grade accuracy as a re-grader. The model needs source context to assign grades correctly.

---

## 2026-03-14: ML Classification Filter (BioLORD + Logistic Regression)

### Approach

The LLM-based post-filter (qwen3:8b) failed because it said YES to everything. Instead of using another LLM, train a lightweight sklearn classifier on BioLORD-2023 embeddings to distinguish true recommendations (TP) from non-recommendations (FP).

### Implementation

1. **Training data generation** (`scripts/generate_classifier_data.py`): Run baseline extraction on all 15 guidelines, match to GT at 0.65 threshold, label matched=TP (1), unmatched=FP (0). Result: 168 examples (91 TP, 77 FP) — NI9RV3E7 skipped (0 text extractions).

2. **Classifier** (`extraction/recommendation_classifier.py`):
   - `LogisticRegression(C=1.0, class_weight='balanced')` on 768-dim BioLORD embeddings
   - Leave-one-guideline-out CV (LOGO-CV) for honest evaluation
   - Probability threshold=0.30 (conservative, removes only high-confidence non-recs)
   - Pipeline position: after verify, before post_filter/normalize

3. **Pipeline integration**: `ml_filter`, `classifier_path`, `ml_filter_threshold` params added to `extract_guideline()`, `self_consistency_extract()`, `auto_vision_extract_guideline()`

4. **CLI**: `--ml-filter`, `--classifier-path`, `--ml-filter-threshold`

5. **A/B configs**: `ml_filter` (priority=17), `sc_adaptive_ml_filter` (priority=18)

### LOGO-CV Results (threshold=0.30)

| Metric | Baseline | ML Filter | Delta |
|--------|----------|-----------|-------|
| Avg F1 | 0.707 | 0.732 | **+0.025** |
| Avg P | 0.608 | 0.646 | **+0.038** |
| Avg R | 0.936 | 0.935 | -0.001 |
| Avg Grade | 0.881 | 0.936 | +0.055 |

### Per-Guideline Detail (LOGO-CV, threshold=0.30)

| Guideline | Base F1 | ML F1 | ΔF1 | Kept |
|-----------|---------|-------|-----|------|
| ICU:10_1007_s00134-024-07369-9 | 0.579 | 0.880 | **+0.301** | 14/23 |
| ICU:10_1007_s00134-025-08058-x | 0.667 | 1.000 | **+0.333** | 3/6 |
| ACP:89499SID | 0.667 | 1.000 | **+0.333** | 1/1 |
| ACP:8J2P9MD8 | 0.667 | 0.800 | +0.133 | 6/7 |
| ICU:10_1007_s00134-025-07840-1 | 0.286 | 0.353 | +0.067 | 14/14 |
| ACP:WND8NBNA | 0.667 | 0.714 | +0.048 | 9/9 |
| ERS:BDYDTUHA | 0.699 | 0.686 | -0.013 | 42/43 |
| ERS:CMCZFLU4 | 0.690 | 0.488 | -0.202 | 29/29 |

### Threshold Sweep

| Threshold | Avg F1 | Avg P | Avg R | ΔF1 vs Base |
|-----------|--------|-------|-------|-------------|
| 0.25 | 0.691 | 0.585 | 0.936 | -0.016 |
| **0.30** | **0.732** | **0.646** | **0.935** | **+0.025** |
| 0.35 | 0.717 | 0.652 | 0.885 | +0.010 |
| 0.40 | 0.642 | 0.582 | 0.791 | -0.065 |

**Verdict: KEPT.** +0.025 F1 with essentially zero recall cost. Best improvements on ICU guidelines with high FP rates. Threshold 0.30 is optimal — more aggressive thresholds start hurting recall.

---

## 2026-03-15: SC Adaptive + ML Filter Combination (NEW BEST)

### Hypothesis

SC Adaptive (F1=0.783) and ML filter (F1=0.732) target different FP types — consensus voting vs. embedding-based classification. Combining them could be additive.

### Results (15-dataset benchmark, excluding NI9RV3E7)

| Metric | Baseline | SC Adaptive | **SC + ML Filter** | Δ vs SC | Δ vs Baseline |
|--------|----------|-------------|-------------------|---------|---------------|
| Avg F1 | 0.707 | 0.783 | **0.860** | **+0.077** | **+0.153** |
| Avg P | 0.608 | 0.788 | **0.937** | +0.149 | +0.329 |
| Avg R | 0.936 | 0.841 | 0.818 | -0.023 | -0.118 |
| Avg Grade | 0.881 | 0.892 | 0.861 | -0.031 | -0.020 |

### Per-Guideline Comparison (SC Adaptive vs SC+ML)

| Guideline | SC F1 | SC+ML F1 | ΔF1 | ML Removed | Key Change |
|-----------|-------|----------|-----|------------|------------|
| ICU:10_1007_s00134-025-07840-1 | 0.333 | 0.857 | **+0.524** | 3/7 | P: 0.22→0.75 |
| ACP:89499SID | 0.667 | 1.000 | +0.333 | 0/1 | SC sampling |
| ACP:48AJE2AR | 0.667 | 1.000 | +0.333 | 0/2 | SC sampling |
| ACP:PAEHSPH3 | 0.857 | 1.000 | +0.143 | 1/4 | Filter removed 1 FP |
| ICU:10_1007_s00134-025-08058-x | 0.857 | 1.000 | +0.143 | 2/5 | P: 0.75→1.00 |
| ACP:WND8NBNA | 0.889 | 1.000 | +0.111 | 0/5 | SC sampling |
| ERS:CMCZFLU4 | 0.690 | 0.800 | +0.110 | 2/15 | P: 0.59→0.77 |
| ICU:10_1007_s00134-024-07369-9 | 0.692 | 0.778 | +0.086 | 5/12 | P: 0.60→1.00 |
| ERS:BDYDTUHA | 0.592 | 0.604 | +0.012 | 0/36 | Minimal change |
| ACP:8V9WED94 | 0.800 | 0.800 | 0.000 | 0/2 | No change |
| ACP:8J2P9MD8 | 1.000 | 1.000 | 0.000 | 0/4 | No change |
| ACP:XLMXNL32 | 0.923 | 0.727 | -0.196 | 0/5 | SC sampling variance |
| ACP:XK8ZAXYM | 1.000 | 0.800 | -0.200 | 0/2 | SC sampling variance |
| ACP:XBAJSPZE | 1.000 | 0.667 | -0.333 | 0/1 | SC sampling variance |

### Analysis

**ML filter impact is clearest on ICU guidelines** where it removed 5/12, 3/7, and 2/5 candidates — all with precision jumps to 0.75–1.00. These high-FP guidelines benefit most from the BioLORD+LR classifier.

**SC stochastic variance** is significant. Guidelines where ML removed 0 recs still show F1 changes (e.g., 48AJE2AR +0.333, XBAJSPZE -0.333) — these are purely from different SC samples at temp=0.3. On small guidelines (1-3 GT recs), a single rec gained or lost changes F1 dramatically.

**Grade accuracy dropped** from 0.892 to 0.861 (-0.031). The ML filter doesn't change grades, so this is SC sampling variance — different consensus-selected recs may have different grade assignments.

**Verdict: NEW BEST CONFIG.** F1=0.860 (+0.077 over SC Adaptive alone, +0.153 over baseline). Precision 0.937 is near-ceiling. The combination is clearly additive — SC removes FPs via consensus, ML filter removes remaining FPs via embedding classification.

### Key Insights

1. **Small dataset (168 examples) works** because logistic regression has few parameters and BioLORD embeddings are already high-quality. More complex models (fine-tuned transformers) would likely overfit.

2. **LOGO-CV is essential** — training and testing on the same guidelines would be circular. Each guideline is predicted by a model that never saw it, giving honest metrics.

3. **The classifier complements SC Adaptive** — SC improves precision via consensus voting, ML filter removes confident non-recs. They target different FP types. The `sc_adaptive_ml_filter` A/B test config will evaluate this combination.

---

## 2026-03-15: Context-Aware Grading Oracle (CLOSED)

### Hypothesis

The context-free grading oracle (2026-03-13) failed because deepseek-r1:32b had no source text. Fix: use BioLORD similarity to retrieve the top-2 most relevant source pages for each recommendation, then pass them as context to the oracle. Conservative prompt: "keep current values" when unsure.

### Implementation

New function `regrade_with_context()` in `extraction/grading_oracle.py`:
- Encodes all PDF pages with BioLORD (once per guideline)
- For each recommendation: cosine similarity → top-2 pages → assemble context → prompt deepseek-r1:32b
- Parses response with existing `_parse_regrade_response()`, only updates valid values
- CLI: `--context-oracle` flag (mutually exclusive with `--grading-oracle`)
- A/B configs: `context_oracle` (priority=19), `sc_adaptive_context_oracle` (priority=20)

### Results (15-dataset benchmark, excluding NI9RV3E7)

| Metric | Baseline | Context Oracle | SC Adaptive | SC + Context Oracle |
|--------|----------|---------------|-------------|---------------------|
| Avg F1 | 0.707 | 0.690 (-0.017) | **0.783** | 0.773 (-0.010) |
| Avg P | 0.608 | 0.571 | 0.788 | **0.836** |
| Avg R | 0.936 | **0.949** | 0.841 | 0.783 |
| Avg Grade | 0.881 | **0.887 (+0.006)** | 0.892 | 0.877 (-0.015) |
| Avg Level | 0.940 | 0.919 | 0.966 | 0.899 |

### Per-Guideline Detail (context_oracle)

| Guideline | Base F1 | Ctx F1 | ΔF1 | Base Grade | Ctx Grade | ΔGrade |
|-----------|---------|--------|-----|------------|-----------|--------|
| ERS:CMCZFLU4 | 0.690 | 0.741 | +0.051 | 0.700 | **0.800** | **+0.100** |
| ERS:BDYDTUHA | 0.699 | 0.797 | +0.098 | 0.972 | 0.957 | -0.015 |
| ACP:XLMXNL32 | 0.750 | 0.667 | -0.083 | 0.833 | 0.833 | 0.000 |
| ICU:10_1007_s00134-024-07369-9 | 0.579 | 0.917 | +0.338 | 1.000 | 1.000 | 0.000 |

### Per-Guideline Detail (sc_adaptive + context_oracle)

| Guideline | SC F1 | SC+Ctx F1 | ΔF1 | SC Grade | SC+Ctx Grade | ΔGrade |
|-----------|-------|-----------|-----|----------|--------------|--------|
| ERS:CMCZFLU4 | 0.870 | 0.818 | -0.052 | 0.900 | 0.778 | -0.122 |
| ERS:BDYDTUHA | 0.674 | 0.559 | -0.115 | 0.944 | 0.923 | -0.021 |
| ACP:WND8NBNA | 0.667 | 0.889 | +0.222 | 1.000 | 0.750 | -0.250 |
| ACP:89499SID | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |

**Verdict: CLOSED.** The context oracle produced mixed results. Standalone, it improved CMCZFLU4 grade (0.70→0.80, the target) and F1 on some guidelines, but overall grade accuracy only improved marginally (+0.006) and F1 dropped (-0.017). When combined with SC Adaptive, the oracle hurt both F1 (-0.010) and grade accuracy (-0.015) — it overwrites correct SC-voted grades with worse ones.

**Key insight:** Adding source context helped CMCZFLU4 grades specifically, but deepseek-r1:32b still has poor calibration across grading schemes. The oracle occasionally overwrites correct values even with context. SC Adaptive alone (F1=0.783, Grade=0.892) remains the best overall config. Grade accuracy as an extractor (0.98) does not reliably transfer to a re-grading role.

---

## M1 real run: v0 snapshot of the 12 labelled guidelines (2026-10-01)

Snapshot 1, version `9e9b07edbbe4` (`configs/v0.json`: single pass, qwen3:14b digest `bdbd181c33f2`, thinking off, prompt v0, 3 few-shot, BioLORD rev `167aab52`), code `bc01c01`. Published, 12/12 active runs, no exclusions, no failures, no resume needed.

### Validation (micro over 12 guidelines, 95% cluster-bootstrap CI; gate NO_BASELINE)
| Metric | Value |
|--------|-------|
| Graded F1 | 0.503 [0.400, 0.623] |
| Precision / Recall | 0.336 / 1.000 |
| Strength / certainty accuracy | 0.955 / 0.909 |
| Combined accuracy | 0.886 [0.816, 0.972] |
| Ungraded P / R | 0.100 / 0.500 |

Per guideline, every GT recommendation is found (FN = 0 on all 12). The false positives cluster in a few guidelines: M22-2056 (15), ANNALS-24-03095 (14), M22-1034 and ANNALS-24-01052 (12 each), M23-2788 (10). Spot check: near-identical consecutive rows that BioLORD dedup at 0.9 keeps. Not comparable to the legacy 0.707: that was a macro average over 14 guidelines incl. ERS, with normalization.

### Runtime (thinking off)
- Per guideline: median 26 s, max 75 s (M22-2056); total 5.7 min for 12.
- Projected ~0.7 h for ~100 guidelines. The ≤ 3 nights M2 budget is not a constraint.
- `think=False` took effect: 0 of 189 calls contain `<think>`, mean 51 eval tokens per call.

### Headline (`out/headline.csv`)
| Society | Guidelines | Recs | Ungraded share | % strong | % against | % high/moderate |
|---------|-----------|------|----------------|----------|-----------|-----------------|
| ACP | 9 | 94 | 0.0 | 40.4 | 5.3 | 69.1 |
| ESICM | 3 | 47 | 21.3 | 0.0 | 5.4 | 33.3 |
| ALL | 12 | 141 | 7.1 | 29.0 | 5.3 | 59.2 |

Counts include the false positives above, so these percentages aren't fit for interpretation yet.

### Issues
- M19-3602: pypdf "invalid code lengths set" / bad float warnings; recovered, all 4 GT recs found.
- `status` low-recall flags (M21-2710, M20-7533, M20-7844, 08058-x) are false alarms: those guidelines have 1–4 GT recs. The society-median rule misfires on small guidelines.
- ESICM 07840-1: ungraded share 0.56 (9 ungraded FPs), from harmonization reading v0 output text.

### For M2
Precision is the problem, not recall or runtime. The SC + ML filter and dedup candidates target exactly this.

## Model swap: v0 on qwen38-64k (2026-10-02)

Snapshot 2, version `e50b2250a148` (`configs/v0_qwen38-64k.json`: v0 with only the model changed), model `qwen38-64k:latest@12af24e3ee72` (qwen3.8 27B, Q4_K_M, 64k context), thinking off. 12/12 succeeded. Validated against published snapshot 1 (`qwen3:14b@bdbd181c33f2`): gate **PASS**. Not published.

| Metric | qwen3:14b (snap 1) | qwen38-64k (snap 2) | Diff [95% paired CI] |
|--------|-------------------|---------------------|----------------------|
| Graded F1 | 0.503 [0.400, 0.623] | **0.854** [0.784, 0.933] | +0.351 [0.218, 0.489] |
| Precision | 0.336 | **0.844** | |
| Recall | **1.000** | 0.864 | |
| Strength / certainty acc | 0.955 / 0.909 | 0.947 / 0.921 | |
| Combined acc | 0.886 | 0.895 | +0.008 [-0.076, 0.088] |
| Ungraded P / R | 0.10 / 0.50 | n/a / 0.00 | |
| Median / max s per guideline | 26 / 75 | 33 / 55 | |
| Total for 12 | 5.7 min | 7.6 min | |

Per guideline (TP/FP/FN, 14b → 38-64k): FPs fall from 87 to 7 in total. Recall losses: ESICM 07369-9 (11/11/0 → 7/0/4), M20-7533 (2/0/0 → 1/0/1), M22-2056 (5/15/0 → 4/2/1). Thinking off held: 0/189 `<think>`, 26 eval tokens per call.

Takeaway: the larger model fixes most of v0's precision problem at about the same speed; recall now limits. ESICM part 1 loses 4 of 11 recommendations.

## Thinking on: qwen38-64k (2026-10-02)

Snapshot 3, version `1fd75aea8f09` (`configs/v0_qwen38-64k_think.json`: snapshot 2's config with thinking on), model `qwen38-64k:latest@12af24e3ee72`. 12/12 succeeded. Gate vs published snapshot 1: PASS. Paired vs snapshot 2 (thinking off): **FAIL**. Not published.

| Metric | qwen38-64k off (snap 2) | qwen38-64k on (snap 3) | On − off [95% paired CI] |
|--------|------------------------|------------------------|--------------------------|
| Graded F1 | **0.854** | 0.769 [0.676, 0.855] | −0.085 [−0.154, −0.037] |
| Precision / Recall | **0.844** / 0.864 | 0.667 / **0.909** | |
| Strength / certainty acc | 0.947 / 0.921 | **0.975 / 0.950** | |
| Combined acc | 0.895 | **0.925** [0.862, 1.000] | +0.030 [−0.036, +0.117] |
| Median / max s per guideline | 33 / 55 | 122 / 267 | |
| Total for 12 / eval tokens | 7.6 min / 4.9k | 28.5 min / 98k | |

Per guideline (TP/FP/FN, off → on): ESICM part 1 recovers fully (7/0/4 → 11/7/0), but FPs rise from 7 to 20 overall (ESICM parts 1–3, M23-2788, M19-3602). Thinking also loses 1 rec each on ANNALS-24-01052 and ANNALS-24-03095. M20-7533 and M22-2056 miss 1 rec in both modes.

Takeaway: thinking trades precision for recall and is 3.7× slower; net F1 is clearly worse. Grade accuracy is slightly better (+0.03), but the CI includes 0. Keep thinking off. Thinking text is not stored: the client keeps only Ollama's `response`, so only its token count is in `eval_tokens`.

## Published: snapshot 2, qwen38-64k thinking off (2026-10-02)

Snapshot 3 rejected ("thinking on: graded F1 0.769 vs 0.854 …, 3.7x slower"). Snapshot 2 published against snapshot 1 (gate PASS). Snapshot 1 stays PUBLISHED in history; the latest published one is current.

Headline now vs ground truth (graded rows, mapped axes):
| | % strong ACP / ESICM | % high/moderate ACP / ESICM |
|---|---|---|
| Ground truth (12 labelled) | 21.4 / 0.0 | 25.0 / 18.8 |
| Snapshot 2 (published) | 22.6 / 0.0 | 32.3 / 14.3 |
| Snapshot 1 (v0) | 40.4 / 0.0 | 69.1 / 33.3 |

v0's percentages were driven by its false positives; snapshot 2 is within a few points of GT.

## Error analysis: snapshot 2 (qwen38-64k, thinking off) (2026-10-02)

13 errors: 6 FN, 7 FP. Method: match each GT row, find its source page, compare with that page's stored LLM output; replay parse + dedup offline from stored `llm_call` rows (replay reproduces the stored score exactly).

### All 6 misses are deduplication, not extraction
The model output every one of them; BioLORD semantic dedup (threshold 0.9) then merged it with a sibling recommendation that differs only in population or comparator:
| Guideline | Lost (GT) | Merged into (similarity) |
|-----------|-----------|--------------------------|
| ESICM 07369-9 | crystalloids vs albumin, *in general* | … *with acute respiratory failure* (0.92) |
| ESICM 07369-9 | crystalloids vs albumin, *with sepsis* | balanced vs saline, *with sepsis* (0.90) |
| ESICM 07369-9 | saline vs *albumin*, TBI | saline vs *balanced*, TBI (0.96) |
| ESICM 07369-9 | balanced vs saline, *in general* | … *with kidney injury* (0.94) |
| ACP M20-7533 | HFNO vs *NIV* (Rec 1a) | HFNO vs *conventional O2* (Rec 1b) (0.90) |
| ACP M22-2056 | *combination* therapy (Rec 1b) | *monotherapy* (Rec 1a) (0.93) |

No miss involves a table: on these 12 guidelines table-first (T5) has nothing to recover.

Dedup replay on stored output (12 guidelines, paired F1 diff vs stored):
| Dedup | Snap 2 (qwen38-64k) P / R / F1 | Δ F1 [CI] | Snap 1 (qwen3:14b) F1 |
|-------|-------------------------------|-----------|------------------------|
| BioLORD 0.90 (current) | 0.844 / 0.864 / 0.854 | — | 0.503 |
| BioLORD 0.95 | 0.843 / 0.977 / 0.905 | +0.051 [−0.012, +0.112] | 0.451 |
| BioLORD 0.97 | 0.830 / 1.000 / 0.907 | +0.053 [−0.031, +0.131] | 0.447 |
| exact only (legacy) | 0.786 / 1.000 / 0.880 | +0.026 [−0.065, +0.116] | 0.438 |

The best threshold depends on the model. qwen3:14b needs aggressive dedup to hide its paraphrased FPs; qwen38-64k doesn't produce those, so dedup only causes damage. The gain is chosen on the same 12 guidelines and its CI includes 0, but the mechanism is direct.

### The 7 false positives
| Count | Cause | Example |
|-------|-------|---------|
| 2 | **GT omission**: real graded recs missing from the labels | ESICM 07840-1 p9: hemorrhagic shock after penetrating/blunt trauma, "(conditional recommendation, moderate certainty of evidence)". GT for this guideline has 3 rows, no trauma recs. |
| 2 | Restatement in rationale text | ANNALS-24-01052 p6: "the CGC suggests that clinicians … use a β-blocker …" |
| 2 | Sub-bullets of a multi-option rec extracted separately | M22-2056: "Switching to or augmenting with cognitive behavioral therapy" |
| 1 | Rationale sentence | M19-3602 p10: "clinicians should avoid prescribing these therapies …" |

Corrected for the GT omission, snapshot 2 precision is 40/45 = 0.889 (not 0.844).

## Dedup 0.97: qwen38-64k (2026-10-02)

Snapshot 4, version `3b2c29833235` (`configs/v0_qwen38-64k_dedup097.json`: snapshot 2's config, `dedup_threshold` 0.9 → 0.97), `qwen38-64k:latest@12af24e3ee72`. 12/12 succeeded, 7.7 min. Gate vs published snapshot 2: **PASS**. Matches the offline replay exactly. Not published.

| Metric | Snap 2 (0.90, published) | Snap 4 (0.97) | Diff [95% paired CI] |
|--------|--------------------------|---------------|----------------------|
| Graded F1 | 0.854 | **0.907** [0.832, 0.966] | +0.053 [−0.031, +0.131] |
| Precision / Recall | 0.844 / 0.864 | 0.830 / **1.000** | |
| Strength / certainty acc | 0.947 / 0.921 | 0.977 / 0.955 | |
| Combined acc | 0.895 | **0.932** | +0.037 [+0.002, +0.102] |

Changed guidelines (TP/FP/FN): ESICM 07369-9 7/0/4 → 11/0/0; M20-7533 1/0/1 → 2/0/0; M22-2056 4/2/1 → 5/2/0; ANNALS-24-03095 and M23-2788 +1 FP each. All 6 misses recovered for 2 FPs. The combined-accuracy CI excludes 0: the recovered recs are graded correctly. Threshold chosen on the same 12 guidelines.

**Published 2026-10-03:** snapshot 4 is the current version (used by `add`, baseline for the next gate). Headline: ACP 35 recs, 22.9% strong, 31.4% high/moderate; ESICM 18 recs, 0.0% strong, 22.2% high/moderate (GT: 21.4 / 25.0 and 0.0 / 18.8).

## Ground truth correction: ESICM part 2 (2026-10-03)

Added the 2 hemorrhagic-shock recommendations (penetrating trauma: conditional, moderate; blunt trauma: conditional, low) to `intensive_care_medicine/10.1007_s00134-025-08058-x.xlsx`, sheet rows 8–9 (end of the 07840-1 group), verified by the user against the PDF. Backup: `….xlsx.bak-2026-10-03`. Labelled set: 46 → 48 rows (ICU 17 → 19).

Rescored with the corrected GT:
| Snapshot | P | R | F1 | Combined acc |
|----------|---|---|----|--------------|
| 1 qwen3:14b | 0.351 | 1.000 | 0.520 | 0.891 |
| 2 qwen38-64k, dedup 0.90 | 0.889 | 0.870 | 0.879 | 0.900 |
| 3 qwen38-64k, thinking (rejected) | 0.700 | 0.913 | 0.792 | 0.929 |
| **4 qwen38-64k, dedup 0.97 (published)** | 0.868 | 1.000 | **0.929** [0.867, 0.978] | 0.935 |

Side effect: the few-shot pool comes from the same GT, so its hash changed. Snapshot 4's version no longer matches the live artifacts (`ArtifactMismatchError: few_shot_pool_sha256`). `add` and `snapshot --resume` with that version are refused until a new snapshot is built.

## Rebuild on corrected GT: snapshot 5 published (2026-10-04)

Snapshot 5, version `113889035d33`: same config as snapshot 4 (`configs/v0_qwen38-64k_dedup097.json`), new version id only because the few-shot pool hash changed with the GT fix. 12/12 succeeded, 7.6 min. Gate vs snapshot 4: PASS, identical scores (F1 0.929, P 0.868, R 1.000, combined 0.935) and identical per-guideline counts. The sampled few-shot examples are the same in 12/12 guidelines. Published; artifacts match the live machine, so `add` is unblocked.

## M2 closed: v1 chosen (2026-10-04)

Last candidate, SC + ML filter on qwen38-64k (snapshot 6, `configs/sc_ml_qwen38-64k.json`: 3 samples, T 0.3, adaptive consensus, clustering at 0.97, ML filter 0.3). Gated vs snapshot 5: PASS (drop not below −0.02 with confidence), but worse:
| Metric | Snap 5 single pass | Snap 6 SC + ML | Diff [95% paired CI] |
|--------|-------------------|----------------|----------------------|
| Graded F1 | **0.929** | 0.882 [0.800, 0.952] | −0.047 [−0.095, −0.009] |
| P / R | **0.868 / 1.000** | 0.804 / 0.978 | |
| Combined acc | 0.935 | 0.933 | −0.001 |
| Median s / guideline | **33** | 103 | 2.8× slower |

Changed (TP/FP/FN): ESICM 07840-1 4/0/0 → 3/0/1 (a real trauma rec filtered), +1 FP on 08058-x and M22-2056, +2 on M23-2788. The classifier was trained on qwen3:14b extractions of these guidelines, so even in-sample it doesn't transfer. Rejected.

All candidates on the corrected GT (48 rows):
| Snap | Candidate | F1 | P | R | s/guideline | Outcome |
|------|-----------|----|---|---|-------------|---------|
| 1 | v0 legacy (qwen3:14b) | 0.520 | 0.351 | 1.000 | 26 | superseded |
| 2 | qwen38-64k, dedup 0.90 | 0.879 | 0.889 | 0.870 | 33 | superseded |
| 3 | + thinking on | 0.792 | 0.700 | 0.913 | 122 | rejected |
| 4/5 | **qwen38-64k, dedup 0.97 = v1** | **0.929** | 0.868 | 1.000 | 33 | **published** |
| 6 | v1 + SC + ML filter | 0.882 | 0.804 | 0.978 | 103 | rejected |
| — | table-first (T5) | — | | | | dropped: no table-related misses |
| — | FT v3 (T9) | — | | | | deferred: qwen3:14b fine-tune, 0.808 in-sample |

Caveat for the paper: every choice (model, thinking, dedup threshold) was made on the same 12 guidelines it is scored on. Spot-check v1 on unlabelled corpus output in M3.

## M3: corpus, snapshots 7 and 8, v2 published (2026-10-06)

### Corpus
PubMed search 2026-10-04 (`scripts/search_guidelines.py`, queries in its `QUERIES`): 426 hits (78 ESICM, 145 SCCM, 203 ACP) in `corpus/candidates.csv`, each with a decision. Included: **73 guidelines** (17 ESICM, 36 SCCM, 20 ACP), 65 current editions, 8 superseded. Excluded by reason: not_guideline 209, derivative 82, not_graded 39, duplicate_publication 11, other_society 10, out_of_range 2. Decisions drafted from titles, PubMed types and the PDFs; reviewed for GRADE wording (all included PDFs grade); the 2024 positioning/mobilisation guideline was moved out (German AWMF, not ESICM).

### PDFs
74 needed, 74 on disk. 46 (17 ACP, 29 SCCM) had to be downloaded by hand: Annals and LWW serve a Cloudflare human check even on an institutional network. Unpaywall, Europe PMC and Semantic Scholar added 1 (PADIS 2018). Tooling: `fetch_pdfs.py` reports `blocked`; `import_pdfs.py missing --open` / `import`. Scalable acquisition is a TODO before M6.

### Snapshot 7: v1 on the corpus (rejected)
73/73 succeeded, 0 failed or empty, ~95 min (median ~45 s; SSC guidelines ~6 min). Validate vs snapshot 5: PASS, but **F1 0.902** (P 0.821, R 1.000) vs 0.929, combined accuracy unchanged 0.935.

Cause: the upload replaced 10 of the 12 labelled PDFs (new Annals downloads; the HAL author manuscript for ESICM 07840-1). Exactly those 10 had different prompts; the 2 unchanged PDFs reproduced snapshot 5 call for call. In the new Annals PDFs the abstract and body copies of a recommendation differ in broken letter spacing ("modific a t i o n si na d u l t sw i ht y p e2"), ligatures and a trailing "(strong recommendation; high-certainty evidence)", so dedup at 0.97 keeps both. Fixes: runs now store the PDF hash (migration 4; backfilled), `status`/`publish` flag changed PDFs.

### Snapshot 8: v2 (published)
v2 = v1 + `dedup_key: alphanumeric` (`configs/v2_qwen38-64k_alnum.json`, version `39ac17093b65`): strip the trailing grade from the text, exact-match on letters and digits. All 1,754 LLM outputs identical to snapshot 7; only dedup differs.

| Metric | Snap 7 (v1) | Snap 8 (v2) | Diff [95% paired CI] |
|--------|-------------|-------------|----------------------|
| Graded F1 | 0.902 | **0.939** [0.867, 0.989] | +0.037 [0.000, 0.083] |
| Precision / Recall | 0.821 / 1.000 | 0.885 / 1.000 | |
| Combined grade accuracy | 0.935 | 0.935 | 0.000 |

On snapshot 5's stored outputs (old PDFs) the same change gives 0.929 → 0.958. v2 removes 34 of 1,546 corpus rows; all 34 checked by hand are duplicates. Gate vs snapshot 5: PASS. Snapshot 7 rejected, snapshot 8 published. Choice again made on the 12 labelled guidelines.

### Harmonization
Snapshot 8 left 3.1% (SCCM) and 4.7% (ACP) of graded recs with unmapped strength (> 2% trigger): "strong/for" labels, "Ungraded"/"good practice"/"Insufficient Evidence" counted as graded, "very low to low". Fixed on read; now SCCM 1.5%, ACP 0%. The rest are mostly the model copying the prompt template ("strong/conditional [for/against]", 23 rows), correctly unmapped. Validation unchanged.

### Headline (snapshot 8, current editions)
| Society | Guidelines | Recs | Ungraded % | Strong % | Against % | High/moderate % |
|---------|-----------|------|------------|----------|-----------|-----------------|
| ACP | 18 | 64 | 0.0 | 31.2 | 12.5 | 46.9 |
| ESICM | 16 | 246 | 10.2 | 18.2 | 15.5 | 29.0 |
| SCCM | 31 | 796 | 2.6 | 19.8 | 19.7 | 21.3 |
| ALL | 65 | 1106 | 4.2 | 20.2 | 18.3 | 24.5 |

SCCM accuracy is unvalidated (no labelled SCCM guideline). `publish` regenerates all 16 outputs byte-identically from the DB; the dashboard serves from `out/dashboard`.

### Triage
`status` outliers are mostly small guidelines (rapid practice guidelines, 1–5 recs) and rationale restatements. Probable real miss: ERC-ESICM temperature control 2022 (1 extracted, ~3 in the PDF). Spot check next.

## M3 spot check: v2 on unlabelled guidelines (2026-10-07)

Spec §7. Sample (seed 0, 3 current unlabelled guidelines per society) plus SSC 2026 by hand (current edition; replaces SSC 2021 in the spec). Snapshot 8 output against each guideline's own recommendation list (summary table or numbered/boxed statements), graded only; one reviewer pass per guideline, every error class re-checked against the PDF. Counts only, no GT written (Decision 3); per-guideline JSON in `out/spotcheck_m3/` (not committed).

| Guideline | TP | FP | FN | Strength ok | Certainty ok |
|-----------|----|----|----|-------------|--------------|
| ACP hypertension ≥60 2017 | 3 | 1 | 0 | 3/3 | 3/3 |
| ACP type 2 diabetes 2017 | 2 | 0 | 0 | 2/2 | 2/2 |
| ACP testosterone 2020 | 3 | 0 | 0 | 3/3 | 3/3 |
| ESICM early enteral nutrition 2017 | 24 | 16 | 0 | 24/24 | 24/24 |
| ERC-ESICM post-resuscitation 2025 | 30 | 2 | 1 | 28/30 | 29/30 |
| ESICM shock and monitoring 2025 | 10 | 5 | 0 | 10/10 | 10/10 |
| SCCM glycemic control 2024 | 6 | 1 | 0 | 6/6 | 6/6 |
| SCCM family-centered care 2024 | 15 | 9 | 0 | 13/15 | 15/15 |
| SCCM/ASPEN pediatric nutrition 2017 | 21 | 10 | 4 | 20/21 | 21/21 |
| SSC adult 2026 | 98 | 8 | 0 | 97/98 | 95/98 |

| Society | P | R | F1 | Strength | Certainty |
|---------|---|---|----|----------|-----------|
| ACP | 0.889 | 1.000 | 0.941 | 8/8 | 8/8 |
| ESICM | 0.736 | 0.985 | 0.842 | 62/64 | 63/64 |
| SCCM | 0.833 | 0.972 | 0.897 | 136/140 | 137/140 |
| **All** | **0.803** | **0.977** | **0.881** | 206/212 (0.972) | 208/212 (0.981) |

Out of sample, recall and grade accuracy hold (labelled set: R 1.000, combined 0.935); precision drops (0.885 → 0.803). SCCM, unvalidated until now, sits between ACP and ESICM.

Error causes (52 FP, 5 FN):
- **Restatements, 38 FP.** A recommendation copied from the summary table and again from the body or rationale, worded differently enough to escape dedup 0.97 (e.g. body drops "critically ill adult patients"). Body copies often carry a guessed certainty (enteral nutrition: "low" for the table's "very low").
- **Ungraded statements extracted as graded, 10 FP.** Good/best-practice and insufficient-evidence statements given a strength (SSC 5, family care 2, ERC 2, ACP 1). Ungraded recall is low overall: 16 of 160 ungraded statements extracted as ungraded, all in the shock guideline.
- **Superseded recommendations from a comparison table, 4 FP.** ESICM shock 2025 Table 1 sets the 2014 consensus beside 2025; the columns interleave in the PDF text and the model extracted 2014 statements with an invented "Strong/High".
- **Misses, 5 FN.** 3 where one extracted row covers several statements of one numbered recommendation (pediatric nutrition), 2 body bullets not extracted (ERC hyperoxaemia, pediatric Harris-Benedict).
- **Grade errors.** Family care: 2 "we make no suggestion for or against" statements labelled "Conditional Against" by the model and kept as against. SSC: 3 recommendations print a certainty per comparator; the model took the first.

Reviewer counting choices move the numbers a little (pediatric nutrition by numbered recommendation: TP 17, FP 14, FN 0; ERC insufficient-evidence items as no recommendation: TP 28, FP 4). Single reviewer, no second annotator: report as a spot check, not a validation.

## M3 spot check: headline bias (2026-10-13)

Do the spot-check errors move the headline? Same 10 guidelines, same metric definitions as `analytics._aggregate`, pooled per recommendation. Published = all snapshot 8 rows (reproduces `out/tables/guidelines.csv`: 280 rows, 264 graded); correct = rows matched to a reference statement; truth = the reference list (217 graded, 160 ungraded). CI: guideline cluster bootstrap, 2000 resamples.

| Metric | Published | Correct | Truth | Published − truth | 95% CI |
|--------|-----------|---------|-------|-------------------|--------|
| % strong | 19.2 | 17.9 | 18.4 | +0.8 | [−2.5, +6.9] |
| % against | 16.9 | 19.8 | 17.5 | −0.6 | [−2.5, +2.4] |
| % high/moderate | 21.4 | 22.6 | 21.5 | −0.1 | [−2.5, +3.5] |
| Ungraded share | 5.7 | — | 42.4 | −36.7 | [−56, −17] |

- Graded metrics: unbiased within 1 point, also without SSC (max +1.1). The errors cancel: restatements raise % against and % high/moderate by ~2 points, ungraded-given-a-grade and the 2014 table lower them. Per society: SCCM within 1 point; ESICM % strong +2.2 (2014 table); ACP +8.3 on 9 recs.
- Ungraded share: unusable. The single-pass prompt asks for best-practice statements but the model rarely emits them (16 of 160, one guideline). The validation gate never saw this (ungraded P/R not gated; labelled set has almost no ungraded rows).

Decision (design Decision 9): M4 draft graded-only; ungraded statements via a separate pass in v3, in parallel (TODOS.md).

## Grade label mapping: statements without a direction (2026-10-13)

Rule: a statement where the panel gives no direction is no recommendation (not graded), whatever grade the model attached. Matched on text, since the model labels them "Conditional Against" or "Conditional". Snapshot 8, 4 rows move out of graded: "no suggestion for or against" (SCCM family care, 2), "unable to issue a recommendation for or against" (SCCM delirium 2025), "cannot suggest for or against" (PCCM 2020, superseded). Headline ALL: graded 1060 → 1057, % against 18.3 → 18.2; SCCM % against 19.7 → 19.5. Validation unchanged (F1 0.939). "We suggest either X or Y" (SCCM CCUS 2025) stays graded: a conditional recommendation that accepts both options.

## Edition links (2026-10-13)

All 8 `supersedes_doi` links checked. ERC-ESICM temperature control 2022 folded into the post-resuscitation topic (the 2025 guideline covers temperature control), so it moves to trends only; current map 64 guidelines. Its Table 2 holds 2 recommendations with no text layer (vector text); v2 extracted 1, from the abstract. Focused updates (PADIS, ultrasound, corticosteroids, NMB) stay current beside the guideline they partly revise: revised questions count twice, ≤ ~25 of 1057 graded recs; Methods states it, recommendation-level supersession in TODOS.md.

## SCCM held-out labels (2026-10-13)

SCCM is 31 of 64 current guidelines and 73% of graded recs, with no labels. Decision: hand-label 3 new SCCM guidelines, out of sample. Random sample (seed 1) of the 27 current SCCM guidelines not in the spot check: safe medication use 2017 (`10.1097/ccm.0000000000002533`), heat stroke 2025 (`10.1097/ccm.0000000000006551`), pediatric end-of-life 2026 (`10.1097/pcc.0000000000003914`); 32 extracted recs. Blind templates (ICU xlsx format + page, rules sheet) in `<data root>/intensive_care_medicine_heldout/`, kept out of the ICU dir so the 12-guideline gate stays comparable. Ungraded statements are labelled too (for v3). `validate` will report this set separately, not gated.

## Corpus search extended before the M4 draft freeze (2026-10-13)

Society-page cross-check (ESICM guidelines page, SCCM list of guidelines, ACP clinical guidelines page) found graded guidelines the M3 title search missed: titles saying "recommendations"/"consensus"/"practice parameters", and co-led guidelines published outside ICM/CCM/PCCM/Annals. Decision: include society-led or co-led guidelines wherever published; widen the search.

`scripts/search_guidelines.py` now runs three queries per society (source column): `pubmed` (M3 title query, unchanged, still 426 hits), `pubmed_broad` (society journal + recommendation/consensus/statement/practice-parameter titles or guideline publication type), `pubmed_society` (society name in title or corporate author, any journal). Re-run with `--until 2026-10-04`: 566 new hits, screened with publication types and abstracts (GRADE/strength wording), then by hand for every society-named or GRADE-flagged hit. Plus 5 society-page-only and 2 same-series rows. 999 candidates, 85 included (+12):

- ESICM: ATS/ESICM/SCCM ARDS ventilation 2017 (AJRCCM, joint SCCM); ERS/ESICM/ESCMID/ALAT HAP/VAP 2017 (ERJ); neuro fluid therapy 2018; ventilation in acute brain injury 2020; resource-limited-settings series (ESICM Global Intensive Care Working Group + MORU, GRADE 1A–2D): infection 2016, malaria/dengue 2016, pediatric sepsis 2016, supportive care 2017, ICU infrastructure 2017.
- SCCM: NCS/SCCM antithrombotic ICH reversal 2016 and its 2026 focused update (Neurocrit Care); ACCM pediatric/neonatal septic shock parameters 2017 (GRADE 1C).
- Co-publications of included guidelines (Resuscitation, ERJ, EJA, JPEN, Acta Anaesthesiol Scand) excluded as duplicate publications; PALICC-2, TAXI, PEACE, Korean/Indian/Chinese societies as other_society (M6 6g); ACP menopause 2026 published after the cutoff (later living update).
- 9 PubMed records without a DOI (family-physician summaries, case reports): not added.

Society attribution for co-led guidelines: the first corpus society in the byline; the others in `joint_with`. 6 PDFs downloaded (open access), 6 still needed by hand (`import_pdfs.py missing`); then `add` with v2 and re-publish before the freeze.

NCS/SCCM antithrombotic ICH 2026 is a focused update that reprints every still-valid 2016 recommendation (its Table 2), unlike PADIS and the other focused updates; linked as superseding 2016 (2016 → trends only). Its Table 1 sets each updated recommendation beside the 2016 version; v2 extracted some of those prior versions (~5–8 of 58 rows), the comparison-table error class seen in the M3 spot check. Kept as extracted; stated as a limitation.

All 12 added guidelines extracted with v2 into snapshot 8 (`add`); `status`: 85/85 runs succeeded, nothing pending, no PDF drift; validation unchanged (F1 0.939). Current map 75 guidelines, 1294 graded recs. Headline ALL: strong 20.2 → 26.1%, against 18.3 → 17.7%, high/moderate 24.5 → 27.5%; ESICM strong 18.2 → 35.5% (resource-limited series and brain-injury ventilation are strong-heavy). Likely over-extraction: ATS/ESICM/SCCM ARDS 2017 gives 15 rows for 6 published recommendations (restatements).

## M4: proof-of-principle analysis package (2026-10-13)

Spec §3–§8. `publish --snapshot 8 --package DIR --analysis-config configs/paper1.json --heldout-labels DIR` opens the frozen DB read-only, audits it against the config (checksums, 85 runs with PDF hashes, 75 current editions, candidates = manifest), scores the held-out SCCM labels (BioLORD on CPU), and writes `out/paper1/` (`docs/paper1/README.md`) besides the standard outputs. Analytics in `evident/paper.py`; held-out and spot-check loaders in `evident/validation.py`; draft skeleton `docs/paper1/draft.md.j2`. Spot-check reference lists committed to `corpus/spotcheck_m3/`.

On the frozen DB: audit passes; the package stops at the held-out labels (3 blank templates), as specified. Table 1 reproduces the headline (ALL 1294 graded; strong 26.1% [20.5, 32.4], against 17.7% [13.0, 22.9], high/moderate 27.5% [22.9, 32.6]). `validate --snapshot 8` on a copy: unchanged (F1 0.939). Spot-check bias now: published − truth strong +0.9, against −1.2, high/moderate 0.0 points; published values moved slightly from the 2026-10-13 bias note (graded 264 → 262) because the no-direction rule came after it.

Captions: each figure states its own caption; the corpus/provenance note appears once. Dashboard: ungraded share hidden (CSV columns kept). SCCM caption changes to "evaluated separately on 3 held-out guidelines" only when the package carries held-out results.

Held-out SCCM (2026-10-19): labels blind, pages left blank (optional now). 2017 safe medication guideline labelled in its own 1/2 + A–D notation, now mapped under GRADE (no snapshot 8 row uses it); its 3 evidence statements are labelled but not scored; one duplicate row removed. Result (n=3, Wilson intervals ignore clustering): TP 29, FP 3, FN 0; P 0.906 [0.758, 0.968], R 1.000 [0.883, 1.000], F1 0.951; strength 29/29, certainty and combined 26/29 (0.897). Ungraded statements: 0/17 recovered (expected for v2). Package from `3738471`, hashes in `freeze.md`. ERS labels deferred to M6 (`TODOS.md`).

## M5: maintenance mode, code (2026-10-26)

Spec `docs/specs/m5-maintenance-spec.md`, runbook `docs/runbook.md`, design Decision 11.

- Live `publish` checks before writing anything: candidates = manifest, coverage (also on re-publish), PDFs as the runs read them, extractor artefacts, search window (`corpus/search.json`), and a verdict for every screened duplicate pair. Any failure leaves `out/` unchanged.
- Duplicate verdicts bind to the judged texts; ledger `corpus/dedup/live_verdicts.csv` seeded from the frozen `paper1_verdicts.csv` (unchanged, hash still matches `configs/paper1.json`). The judge scripts take `--snapshot/--manifest/--ledger/--dry-run`; the consensus merges into the ledger and refuses the frozen file.
- `add`: refuses a changed PDF without `--force` (bytes in the PDF dir untouched), checks curation before extracting, exits 3 when pairs need judging; `--no-publish` stages. `rebind-pdf` accepts new bytes when every rebuilt prompt hash matches the run's (no model call).
- `status --published --check`: the release, even while a candidate builds; exit 1 on a blocker.
- `out/release.json` (schema 1): id = SHA-256 of sources, versions, inputs, counts and output checksums; archived with a DB backup in `artifacts/releases/<id>/`.
- Tests: 71 new, 805 pass (pipeline, CLI, scripts, runbook procedures A–C on a scratch DB).
- On a copy of the live DB: `status --published --check` → only blocker "no release record"; `publish --snapshot 8` → 75 current guidelines, 140 duplicates removed (as M4), release `f0e36aa13d5f…`; repeated → `unchanged`, outputs byte-identical. The live `out/` was not touched.

## M5: first live release (2026-10-27)

Code `375beae`; DB backup `artifacts/backups/evident-2026-10-27-0936.sqlite`.

- `status --published --check` before: only blocker "no release record".
- `publish --snapshot 8` (47 s): 75 current, 10 superseded, 1827 recommendations, 140 duplicates removed, 1687 kept. Release `65bd85c85387…` (`out/release.json`, archived in `artifacts/releases/`). Replaces the package-mode outputs in `out/`.
- Repeated: `unchanged`. `status --published --check`: `check: clean`.

Exit still open: one real guideline published after 2026-10-04, added with runbook A.


## M5: PR #4 review fixes (2026-10-27)

- Strict status blocks on inputs changed since the release (manifest, candidates, search window, verdict ledger, themes, PDF of a released run); code commits alone do not.
- `publish` archives before swapping outputs; the three output dirs swap all or none; `out/release.json` last.
- Every older edition must be superseded explicitly (`supersedes_doi` takes `a;b`). ERC-ESICM 2025 now names temperature control 2022 (was implicit); 75 current, 10 superseded unchanged. Manifest no longer matches `configs/paper1.json` (frozen copy at `paper1-freeze`).
- Tests: 11 new, 816 pass. Live strict status: `changed since release 65bd85c85387: manifest` until republished.
- Release `0c1c6f4868a4…` from code `e7125a1` (backup `artifacts/backups/evident-2026-10-27-1958.sqlite`): 75 current, 10 superseded, 140 duplicates removed; repeated → `unchanged`; `check: clean`.
