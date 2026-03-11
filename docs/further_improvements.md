# Further Improvements for the Extraction Pipeline

Last updated: 2026-03-11. Based on benchmark results and experimentation.

---

## Remaining Open Ideas

### 1. Vision Table Extraction Improvements (NI9RV3E7 — current F1=0.65)

The main remaining opportunity. NI9RV3E7 (217 GT recs) uses vector-rendered tables that require vision LLM extraction. Current best: mistral-small3.2:24b (F1=0.65, P=0.79, R=0.55).

#### 1A. Hybrid Structure+Vision Pipeline (High effort, highest potential)

- Use `img2table` or Microsoft Table Transformer for cell-level bounding box detection
- Apply OpenCV color preprocessing per cell (NI9RV3E7 uses color coding: green=I, yellow=IIa, orange=IIb, red=III)
- Per-cell OCR or vision extraction instead of full-page → less hallucination
- Color detection alone could determine grades without OCR

#### 1B. Table Page Classification (Low effort)

- Currently pdfplumber detects 35 "table pages" but many are definition/prognostic tables, not recommendation tables
- Keyword heuristic ("recommendation", "class", "level of evidence") or color-pattern detection could filter
- Reduces noise pages sent to vision model

#### 1C. Alternative Vision Models

| Model | Size | Why promising |
|-------|------|---------------|
| minicpm-v:8b | 4.9GB | MiniCPM-V 2.6: excellent table extraction in benchmarks |
| llava:13b | 7.4GB | Established vision-language model, less hallucination-prone |

Note: qwen2.5vl:7b crashes with GGML error — unusable.

### 2. Adaptive Similarity Thresholds (Low effort)

Fixed similarity threshold (0.65) is suboptimal. Short recommendations may need lower thresholds. Length-adaptive thresholds or per-dataset calibration could help.

### 3. NuExtract as Alternative Model (High effort)

[NuExtract](https://numind.ai/blog/nuextract-2-0) is a fine-tuned extraction model (0.5B-7B) that outperforms GPT-4o on structured extraction benchmarks. Purpose-built for the task, much smaller and faster. Would need evaluation on medical terminology.

### 4. Better Ensemble Strategies (Medium effort)

Current ensemble (union+dedup) failed because it adds too many false positives. Alternative: **intersection-based ensemble** — only keep recommendations extracted by both models. Or: weighted voting based on model confidence/logprobs. deepseek-r1:32b has the best grade accuracy (0.98) but worst precision — could be used as a grading oracle on qwen3:14b's extractions.

---

## Closed / Completed

| # | Improvement | Status | Outcome |
|---|-------------|--------|---------|
| — | Vision extraction (NI9RV3E7) | **DONE** | F1 0.00→0.65 with mistral-small3.2:24b |
| — | Grading scheme detection | **DONE** | Fixed 0% ERS grade accuracy |
| — | Cross-scheme few-shot fallback | **DONE** | BDYDTUHA F1 +0.06 |
| — | Parser hardening | **DONE** | Zero-cost defensive improvement |
| — | Token overlap verification | **DONE** | Near-zero impact, kept as safety net |
| — | Self-consistency voting | **CLOSED** | -0.05 F1, stochastic runs too similar |
| — | CoT prompt | **CLOSED** | Causes regressions (model overthinks) |
| — | Improved few-shot selection | **CLOSED** | Both stratified and negative examples regress |
| — | JSON structured output | **CLOSED** | Over-extracts badly (avg F1=0.47) |
| — | Two-pass extraction | **CLOSED** | llama3.2 classifier too conservative |
| — | Ensemble (union+dedup) | **CLOSED** | Too many false positives (avg F1=0.42) |
| — | 3-page chunking | **CLOSED** | Hurt ACP precision |
| — | Alias expansion | **DONE** | Included in grading scheme fix |

## Key Insight: The Precision Gap

qwen3:14b's false positives are real text from the PDF (background statements, evidence summaries) misclassified as recommendations — not hallucinations. Approaches that try to verify source grounding (token overlap, self-consistency) don't help. Approaches that try to make the model more conservative (negative examples, CoT) hurt recall more than they help precision. The remaining precision gap requires fundamentally different approaches (fine-tuning, better base models, or semantic classifiers).

---

## References

- NuExtract 2.0: https://numind.ai/blog/nuextract-2-0
- img2table (cell detection): https://github.com/xavctn/img2table
- Microsoft Table Transformer: https://huggingface.co/microsoft/table-transformer-structure-recognition
- bespoke-minicheck (NLI verification): https://huggingface.co/bespoke-stratos/Bespoke-Minicheck-7B
