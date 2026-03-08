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

**Analysis:** Prompt changes helped ERS guidelines (especially CMCZFLU4 +0.27) but hurt ACP precision. The improved prompt is more specific about what constitutes a recommendation, which reduces false positives for ERS but also reduces true positives for ACP where recommendations are less formulaic.

### Phase 2 Benchmark: qwen3:14b few-shot + normalize

| Guideline | Baseline F1 | qwen3:14b F1 | Delta | Grade | Level |
|-----------|------------|-------------|-------|-------|-------|
| ACP:XLMXNL32 | 0.71 | **0.91** | +0.20 | 1.00 | 1.00 |
| ACP:8J2P9MD8 | 0.89 | 0.73 | -0.16 | 0.75 | 1.00 |
| ACP:8V9WED94 | 0.67 | 0.57 | -0.10 | 0.50 | 0.50 |
| ERS:NI9RV3E7 | 0.01 | 0.00 | -0.01 | 0.00 | 0.00 |
| ERS:CMCZFLU4 | 0.48 | 0.13 | -0.35 | 0.00 | 0.00 |
| ERS:BDYDTUHA | 0.62 | 0.58 | -0.04 | 0.89 | 0.89 |

**Analysis:** qwen3:14b achieves perfect precision on XLMXNL32 (F1=0.91) — the best single-guideline result. But it's too conservative on ERS guidelines, particularly CMCZFLU4 (only 3 extractions from 12 GT). The larger model is better at discriminating what is/isn't a recommendation but over-filters on some guidelines.

---

## Key Insights

### NI9RV3E7 is a PDF extraction problem
The 80-page ERS pulmonary hypertension guideline has 217 ground truth recommendations, all presented in **structured tables** within the PDF. The PyPDF text extraction captures surrounding text and table footnotes ("Class of recommendation", "Level of evidence") but not the table rows containing the actual recommendations. No LLM or prompt changes can fix this — it requires a table-aware PDF extraction approach (e.g., camelot, tabula, or vision-based extraction).

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
- Drops rows with invalid "0.0" values (reduces noise in GT data)
- ACP GT normalization: truncation fix ("onditional recommendation"→"Weak For", "trong"→"Strong For")
- Trailing punctuation stripped (e.g., "strong recommendation;" → "strong recommendation")

#### Extended Grade/Level Aliases (`evaluation/grading.py`)
- ESC_ERS: "class 1"→I, "class 2a"→IIa, "ii a"→IIa, level of evidence aliases
- GRADE: confidence-style levels ("low confidence in estimates of effect"→Low)
- GRADE: GT truncation artifacts ("onditional recommendation"→Weak For)
- GRADE: "moderate/low-certainty evidence"→Low

### Benchmark: Corrected Schemes

#### qwen3:8b few-shot + normalize (corrected schemes)

| Guideline | Scheme | Baseline F1 | Fixed F1 | Baseline Grade | Fixed Grade |
|-----------|--------|-------------|----------|----------------|-------------|
| ACP:XLMXNL32 | grade | 0.71 | **0.80** | 1.00 | 0.83 |
| ACP:8J2P9MD8 | grade | 0.89 | 0.62 | 0.75 | **1.00** |
| ACP:8V9WED94 | grade | 0.67 | **0.80** | 0.50 | **1.00** |
| ERS:NI9RV3E7 | esc_ers | 0.01 | 0.00 | 0.00 | 0.00 |
| ERS:CMCZFLU4 | grade | 0.48 | **0.83** | 0.00 | **0.90** |
| ERS:BDYDTUHA | abcd_123 | 0.62 | 0.60 | 0.00 | **0.59** |

#### qwen3:14b few-shot + normalize (corrected schemes)

| Guideline | Scheme | Baseline F1 | Fixed F1 | Baseline Grade | Fixed Grade |
|-----------|--------|-------------|----------|----------------|-------------|
| ACP:XLMXNL32 | grade | 0.71 | **0.91** | 1.00 | **1.00** |
| ACP:8J2P9MD8 | grade | 0.89 | 0.67 | 0.75 | 0.75 |
| ACP:8V9WED94 | grade | 0.67 | 0.67 | 0.50 | 0.50 |
| ERS:NI9RV3E7 | esc_ers | 0.01 | 0.00 | 0.00 | 0.00 |
| ERS:CMCZFLU4 | grade | 0.48 | **0.83** | 0.00 | **1.00** |
| ERS:BDYDTUHA | abcd_123 | 0.62 | **0.68** | 0.00 | **0.91** |

**Key wins:**
- CMCZFLU4: Grade accuracy 0.00 → 0.90-1.00 (was using wrong scheme entirely!)
- BDYDTUHA: Grade accuracy 0.00 → 0.59-0.91
- ACP:8V9WED94: Grade 0.50 → 1.00 (GT data cleaned)
- qwen3:14b: Best BDYDTUHA Grade=0.91, Level=0.80

### New Models Downloaded
- mistral-small3.2:24b (15GB)
- qwen3:30b-a3b (18GB)
- gemma3:27b (17GB)

---

## 2026-03-06: Full Model Comparison (7 models, corrected schemes)

All models benchmarked with few-shot + normalize + corrected grading scheme detection.

### Results: F1 Score (excluding NI9RV3E7 — PDF table problem)

| Model | Size | XLMXNL32 | 8J2P9MD8 | 8V9WED94 | CMCZFLU4 | BDYDTUHA | Avg F1 |
|-------|------|----------|----------|----------|----------|----------|--------|
| qwen3:14b | 9.3GB | **0.92** | **0.89** | 0.80 | **0.87** | 0.68 | **0.83** |
| qwen3:8b | 5.2GB | 0.80 | 0.62 | 0.80 | 0.83 | 0.60 | 0.73 |
| deepseek-r1:32b | 19GB | 0.86 | 0.67 | 0.27 | 0.69 | 0.70 | 0.64 |
| mistral-small3.2:24b | 15GB | **1.00** | 0.67 | 0.80 | 0.47 | 0.44 | 0.68 |
| gemma3:27b | 17GB | 0.43 | 0.80 | 0.44 | 0.49 | **0.76** | 0.58 |
| qwen3:30b-a3b | 18GB | 0.80 | 0.00 | **1.00** | **0.95** | 0.00 | 0.55 |

### Results: Grade Accuracy

| Model | XLMXNL32 | 8J2P9MD8 | 8V9WED94 | CMCZFLU4 | BDYDTUHA | Avg Grade |
|-------|----------|----------|----------|----------|----------|-----------|
| qwen3:14b | 1.00 | **1.00** | 0.50 | 0.80 | 0.91 | **0.84** |
| deepseek-r1:32b | 1.00 | **1.00** | **1.00** | 0.90 | **0.98** | **0.98** |
| qwen3:8b | 0.83 | **1.00** | **1.00** | 0.90 | 0.59 | 0.86 |
| mistral-small3.2:24b | 1.00 | 0.75 | 0.50 | 0.62 | 0.94 | 0.76 |
| gemma3:27b | 1.00 | 0.75 | 0.50 | 0.70 | **1.00** | 0.79 |
| qwen3:30b-a3b | 1.00 | 0.00 | 0.50 | 0.67 | 0.00 | 0.43 |

### Results: Level Accuracy

| Model | XLMXNL32 | 8J2P9MD8 | 8V9WED94 | CMCZFLU4 | BDYDTUHA | Avg Level |
|-------|----------|----------|----------|----------|----------|-----------|
| qwen3:14b | **1.00** | **1.00** | **1.00** | **1.00** | 0.80 | **0.96** |
| qwen3:8b | 0.83 | **1.00** | **1.00** | **1.00** | 0.59 | 0.89 |
| deepseek-r1:32b | **1.00** | **1.00** | **1.00** | 0.80 | 0.82 | 0.92 |
| mistral-small3.2:24b | 0.83 | **1.00** | **1.00** | 0.75 | 0.94 | 0.91 |
| gemma3:27b | 0.83 | **1.00** | **1.00** | **1.00** | 0.78 | 0.92 |
| qwen3:30b-a3b | **1.00** | 0.00 | **1.00** | **1.00** | 0.00 | 0.60 |

### Key Findings

**Best overall model: qwen3:14b** (avg F1=0.83, avg Grade=0.84, avg Level=0.96)
- Highest average F1 by a wide margin
- Perfect recall on 4/5 extractable guidelines
- Best balance of precision and recall
- Smallest model that performs well (9.3GB)

**Best grade accuracy: deepseek-r1:32b** (avg Grade=0.98 on matched recs)
- Perfect or near-perfect grading when it finds the recommendation
- But lower precision (extracts too many false positives, especially 8V9WED94: 13 extracted vs 2 GT)

**Most inconsistent: qwen3:30b-a3b**
- Extreme variance: F1=0.00 on 8J2P9MD8 and BDYDTUHA, but F1=0.95 and 1.00 on others
- MoE architecture may cause instability with the thinking/extraction format

**Perfect extraction: mistral-small3.2:24b on XLMXNL32**
- F1=1.00 (6/6 recs, 0 false positives) — the only perfect extraction across all benchmarks
- But poor recall on ERS guidelines (BDYDTUHA: only 18/60 found)

**NI9RV3E7 remains unsolvable** with current approach (all models 0% or near-0%)
- deepseek-r1:32b extracts the most (F1=0.16) — likely from table footnotes
- gemma3:27b also extracts some (F1=0.12)
- Requires table-aware PDF extraction (Docling)

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

NI9RV3E7 (80-page ERS pulmonary hypertension guideline, 217 GT recs) had 0% extraction with all text-based approaches. Investigation revealed:

- All 217 recommendations live in **tables rendered as vector graphics**
- PyPDF, Docling, and pdfplumber all detect table bounding boxes but extract **zero characters** from within them
- The table cell text simply doesn't exist in the PDF text layer — it's drawn as paths/glyphs

### Approach: Page Image → Vision LLM

1. **pdfplumber** detects which pages contain tables (35 of 80 pages)
2. Full pages rendered as PNG images at 300 DPI
3. **gemma3:27b** (vision-capable) reads each page image and extracts recommendations
4. Results parsed, deduplicated, and optionally combined with text-based extraction

### Implementation

#### New: `extraction/pdf_loader.py`
- `PDFTableImage` dataclass: page_number, table_index, image_bytes
- `load_pdf_table_images()`: uses pdfplumber to find table pages, renders full page as PNG
- Installed: `pdfplumber`, `pytesseract` (tesseract OCR tested but inferior to vision LLM)

#### New: `extraction/vision_extractor.py`
- `extract_tables_with_vision()`: sends page images to vision LLM, parses pipe-delimited output
- `vision_extract_guideline()`: combines vision table extraction with optional text-based extraction
- `_build_vision_prompt()`: scheme-aware prompt with Class/Level value hints
- `_parse_vision_response()`: handles markdown lists, bold, label prefixes

#### Updated: `run_benchmark.py`
- New flags: `--vision`, `--vision-model <model>`
- Vision mode uses `vision_extract_guideline()` instead of `extract_guideline()`

### Results on NI9RV3E7

| Approach | Extracted | F1 | P | R | Grade | Level |
|----------|-----------|------|------|------|-------|-------|
| Text only (best: deepseek-r1:32b) | 62 | 0.16 | 0.35 | 0.10 | 0.68 | 0.18 |
| **Vision (gemma3:27b)** | **209** | **0.27** | **0.27** | **0.26** | **0.84** | **0.83** |
| Vision @0.50 threshold | 209 | 0.43 | 0.44 | 0.42 | 0.69 | — |
| Vision @0.40 threshold | 209 | 0.52 | 0.53 | 0.50 | 0.62 | — |

### Analysis

- Vision extracts 209 recs (vs 217 GT) — nearly complete count
- Grade/level accuracy on matched recs is excellent (0.84/0.83)
- **Low F1 at standard threshold (0.65)** is due to text wording differences:
  - Vision: "RHC is recommended to confirm the diagnosis" (abbreviated from table)
  - GT: "RHC is recommended to confirm the diagnosis of pulmonary arterial hypertension (group 1) and to support treatment decisions" (full text)
- Class distribution mismatch: vision extracts 91 Class III vs GT's 20 (misreads some table structures)
- 3209 raw recs before dedup → 209 after (heavy duplication from overlapping table content)

### Environment Changes

- Installed: `docling` (2.69.1), `pdfplumber` (0.11.8), `pytesseract` (0.3.13), `onnxruntime` (1.19.2)
- Removed: `tensorflow` (2.8.0) — incompatible with upgraded numpy/transformers from docling
- Upgraded: `scikit-learn` (1.1.0 → 1.6.1), `pandas` (1.4.1 → 2.3.3), `transformers` (4.40.0 → 4.57.6)

### What was tried but didn't work

- **Docling** (IBM): Detected 35 tables but extracted 0 cell content (same vector graphics problem)
- **pdfplumber table extraction**: Detected table grids but all cells empty
- **Tesseract OCR**: Got text from images but lost table structure, garbled on colored backgrounds
- **RapidOCR**: Chinese-optimized, failed to detect any English text
- **Table bbox cropping**: pdfplumber detects partial bounding boxes that cut off recommendation text columns — had to switch to full-page rendering

---

## Pending / Not Yet Benchmarked

- [x] JSON structured output (`--json`) — CLOSED: significantly worse (see below)
- [x] Two-pass extraction (`--two-pass`) — CLOSED: mixed results, net negative (see below)
- [x] Ensemble/voting across models — CLOSED: worse F1 due to precision loss (see below)
- [x] Multi-page chunking (`--pages-per-chunk 3`) — CLOSED: tested, hurt ACP precision (too many false positives), no benefit
- [x] Vision model comparison — DONE: mistral-small3.2:24b best vision model (see below)
- [x] Improve vision extraction prompt — DONE: anti-hallucination rules + tighter generation params
- [ ] Post-processing to expand abbreviated vision-extracted recommendations — NOT NEEDED (vision text not expandable from PDF text layer)

---

## 2026-03-08: Benchmark Batch — JSON, Two-Pass, Ensemble, Vision Models

### Step 1: JSON Structured Output (`--json`)

**Command:** `python run_benchmark.py --model qwen3:14b --few-shot-only --json --normalize`

| Guideline | Baseline F1 | JSON F1 | Delta | Extracted | GT | Grade | Level |
|-----------|------------|---------|-------|-----------|-----|-------|-------|
| ACP:XLMXNL32 | **0.92** | 0.40 | -0.52 | 24 | 6 | 1.00 | 1.00 |
| ACP:8J2P9MD8 | **0.89** | 0.67 | -0.22 | 8 | 4 | 1.00 | 1.00 |
| ACP:8V9WED94 | 0.67 | 0.29 | -0.38 | 11 | 3 | 0.50 | 1.00 |
| ERS:NI9RV3E7 | 0.00 | 0.16 | +0.16 | 90 | 217 | 0.44 | 0.28 |
| ERS:CMCZFLU4 | **0.87** | 0.69 | -0.18 | 20 | 12 | 0.73 | 0.91 |
| ERS:BDYDTUHA | **0.68** | 0.64 | -0.04 | 47 | 60 | 0.94 | 0.94 |
| **Average** | **0.83** | **0.47** | **-0.36** | | | 0.77 | 0.86 |

**Verdict: NEGATIVE.** JSON output causes massive over-extraction (e.g., 24 recs for 6 GT on XLMXNL32). The structured format appears to encourage the model to hallucinate more recommendations. Average F1 drops from 0.83 to 0.47.

### Step 2: Two-Pass Extraction (`--two-pass`)

**Command:** `python run_benchmark.py --model qwen3:14b --few-shot-only --two-pass --normalize`

| Guideline | Baseline F1 | Two-Pass F1 | Delta | Extracted | GT | Grade | Level |
|-----------|------------|------------|-------|-----------|-----|-------|-------|
| ACP:XLMXNL32 | **0.92** | 0.00 | -0.92 | 6 | 6 | 0.00 | 0.00 |
| ACP:8J2P9MD8 | 0.89 | 0.89 | 0.00 | 5 | 4 | 0.75 | 1.00 |
| ACP:8V9WED94 | 0.67 | 0.67 | 0.00 | 3 | 3 | 0.50 | 1.00 |
| ERS:NI9RV3E7 | 0.00 | 0.00 | 0.00 | 0 | 217 | 0.00 | 0.00 |
| ERS:CMCZFLU4 | 0.87 | **0.91** | +0.04 | 10 | 12 | 1.00 | 1.00 |
| ERS:BDYDTUHA | **0.68** | 0.34 | -0.34 | 16 | 60 | 1.00 | 1.00 |
| **Average** | **0.83** | **0.47** | **-0.36** | | | 0.54 | 0.67 |

**Verdict: NEGATIVE.** The llama3.2 page classifier is too conservative — it filters out pages containing recommendations on XLMXNL32 (F1=0.00 despite extracting 6 recs) and BDYDTUHA (only 16/60 found). CMCZFLU4 improved slightly (F1=0.91 with perfect grade/level). The 3-page chunking in pass 2 may also contribute to issues. Two-pass needs a better classifier to be viable.

### Step 3: Multi-Page Chunking

Already tested previously: hurt ACP precision. Closed, no further work needed.

### Step 4: Ensemble (qwen3:14b + deepseek-r1:32b)

**Command:** `python run_benchmark.py --model qwen3:14b --few-shot-only --ensemble --normalize`

| Guideline | Baseline F1 | Ensemble F1 | Delta | Extracted | GT | Grade | Level |
|-----------|------------|------------|-------|-----------|-----|-------|-------|
| ACP:XLMXNL32 | **0.92** | 0.55 | -0.37 | 16 | 6 | 1.00 | 0.83 |
| ACP:8J2P9MD8 | **0.89** | 0.32 | -0.57 | 21 | 4 | 1.00 | 1.00 |
| ACP:8V9WED94 | 0.67 | 0.40 | -0.27 | 7 | 3 | 1.00 | 1.00 |
| ERS:NI9RV3E7 | 0.00 | 0.11 | +0.11 | 46 | 217 | 0.57 | 0.29 |
| ERS:CMCZFLU4 | **0.87** | 0.50 | -0.37 | 28 | 12 | 0.80 | 1.00 |
| ERS:BDYDTUHA | 0.68 | 0.63 | -0.05 | 109 | 60 | 0.96 | 0.87 |
| **Average** | **0.83** | **0.42** | **-0.41** | | | 0.89 | 0.83 |

**Verdict: NEGATIVE.** Ensemble achieves perfect recall on ACP but terrible precision — deepseek-r1:32b adds many false positives that exact+semantic dedup can't remove (they're genuinely different wrong recommendations). Grade accuracy is excellent (0.89 avg) when matches are found, but the low F1 makes this impractical. Would need voting/agreement filtering instead of union+dedup.

### Step 5: Vision Model Comparison on NI9RV3E7

**Improved vision prompt:** Added anti-hallucination rules ("Only extract rows VISIBLE in the table", "Do NOT generate recommendations not shown", "Do NOT abbreviate") and tighter generation params (top_p=0.1, repeat_penalty=1.1).

#### NI9RV3E7 Results (auto-vision, text model: qwen3:14b)

| Vision Model | Size | Extracted | Raw | F1 | P | R | Grade | Level | Time |
|-------------|------|-----------|-----|------|------|------|-------|-------|------|
| **mistral-small3.2:24b** | 15GB | **151** | 213 | **0.65** | **0.79** | 0.55 | **0.94** | **0.97** | 731s |
| gemma3:27b | 17GB | 79 | 995 | 0.30 | 0.57 | 0.21 | 0.69 | 0.84 | 1120s |
| qwen2.5vl:7b | 4.7GB | CRASH | — | — | — | — | — | — | — |

#### Full ERS Results (auto-vision with each vision model)

| Vision Model | NI9RV3E7 F1 | CMCZFLU4 F1 | BDYDTUHA F1 | ERS Avg F1 | ERS Avg Grade |
|-------------|-------------|-------------|-------------|------------|---------------|
| **mistral-small3.2:24b** | **0.65** | 0.74 | 0.68 | **0.69** | **0.95** |
| gemma3:27b | 0.30 | 0.59 | 0.68 | 0.52 | 0.83 |
| qwen2.5vl:7b | ERROR | 0.48 | 0.68 | N/A | N/A |

**Key findings:**
- **mistral-small3.2:24b is the best vision model** — dramatically better than gemma3:27b on NI9RV3E7
  - 2.2x better F1 (0.65 vs 0.30)
  - 3.8x better precision (0.79 vs 0.57) — much less hallucination
  - Only 213 raw recs vs gemma3's 995 (much more focused)
  - Grade accuracy 0.94 vs 0.69
- qwen2.5vl:7b crashes with GGML assertion error on vision inference (Ollama backend incompatibility)
- Improved prompt + generation params contributed to better results vs previous gemma3 baseline (0.30 vs 0.27)
- CMCZFLU4 and BDYDTUHA use text-only extraction (no opaque tables detected)

### Summary of All Experiments

| Approach | Avg F1 (excl NI9RV3E7) | Best For |
|----------|----------------------|----------|
| **Baseline (qwen3:14b few-shot + normalize)** | **0.83** | Overall best |
| JSON output | 0.47 | Nothing (over-extracts) |
| Two-pass | 0.47 | CMCZFLU4 only (F1=0.91) |
| Ensemble | 0.42 | Grade accuracy (0.89) |
| Auto-vision (mistral) | 0.69 (all ERS) | NI9RV3E7 (F1=0.65) |

**Recommended configuration:**
- Text extraction: `qwen3:14b --few-shot-only --normalize` (baseline, avg F1=0.83)
- Vision-capable PDFs: add `--auto-vision --vision-model mistral-small3.2:24b`
- NI9RV3E7 specifically: F1 improved from 0.00 (text-only) to 0.65 (auto-vision with mistral)
