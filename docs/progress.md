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
