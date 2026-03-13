# Further Improvements for the Extraction Pipeline

Last updated: 2026-03-13. Based on 15-dataset A/B validation benchmark.

---

## Current Performance Summary

15 datasets (9 ACP + 3 ERS + 3 ICU). Excluding NI9RV3E7 (vision-only):

| Config | Avg F1 | Avg P | Avg R | Avg Grade |
|--------|--------|-------|-------|-----------|
| **SC Adaptive (best F1)** | **0.783** | 0.788 | 0.841 | 0.892 |
| SC Fixed (consensus=2) | 0.750 | 0.820 | 0.766 | 0.891 |
| Baseline (few-shot+norm) | 0.707 | 0.608 | 0.936 | 0.881 |

**Core problem:** Precision is the bottleneck. ~40% of baseline extractions are false positives — real PDF text (background statements, evidence summaries) misclassified as recommendations. Adaptive self-consistency helps precision (+30%) with modest recall trade (-10%).

**Worst performers:**
- ICU guidelines: P=0.17–0.41 (2–6× over-extraction), perfect recall
- CMCZFLU4: Grade accuracy 0.70 (GRADE scheme Strong/Conditional confusion)
- BDYDTUHA: R=0.60 (missing 40% of 60 GT recs), SC makes it worse (R→0.45)
- NI9RV3E7: F1=0.00 text-only, F1=0.65 with vision

---

## Phase 1: Quick Wins

### 1. Fine-tuned Clinical ModernBERT Classifier

[Clinical ModernBERT](https://arxiv.org/abs/2504.03964) (110M–400M params, <1GB VRAM) achieves AUROC=0.977 on clinical text classification. Fine-tune on:
- **Positives:** GT recommendations from all 15 guidelines (~350 examples)
- **Negatives:** False positives logged from benchmark runs (extracted but not matched to GT)

The LLM-based post-filter (qwen3:8b) failed because it said YES to everything. A purpose-built classifier trained on actual false positives could succeed where the generic LLM couldn't discriminate.

- **VRAM:** <1GB, can run on CPU
- **Expected impact:** +0.05–0.10 F1. Purpose-built for the classification problem.
- **Risk:** Needs a labeled dataset of false positives. Can extract these from existing benchmark CSVs.

---

## Phase 2: Medium Effort (3–5 days each)

### 4. NuExtract 2.0 as Alternative Extractor

[NuExtract 2.0](https://numind.ai/blog/nuextract-2-0) (2B/4B/8B) is purpose-built for structured text-to-JSON extraction. Available on Ollama. The 8B model (~7GB VRAM) achieves 73 F-Score on 21 diverse extraction tasks. Has "higher precision than recall" bias.

- **VRAM:** 8B needs ~7GB at Q4_K_M
- **Supports:** In-context learning with 3 examples, JSON schema input
- **Caveat:** 2000-token input limit per call → requires heavy page chunking
- **Expected impact:** Medium — precision-focused by design, but untested on medical recommendation semantics
- **Effort:** Medium (on Ollama, but needs prompt adaptation to its template format)

### 5. deepseek-r1:32b as Grading Oracle

deepseek-r1:32b has the best grade accuracy (0.98) but worst precision. Use it as a second pass: qwen3:14b extracts recommendations, deepseek-r1:32b re-grades them.

- **Expected impact:** +0.05–0.10 grade accuracy (especially CMCZFLU4: 0.70→0.90+)
- **VRAM:** 19GB, must swap with qwen3:14b (sequential, not parallel)
- **Speed:** Adds ~5–10 min per guideline
- **Effort:** Medium (~50 lines, pipeline two model calls)

### 6. DSPy Prompt Optimization

[DSPy](https://dspy.ai/) automates prompt engineering. Uses MIPROv2 optimizer to search for better prompt formulations against your F1 metric. Works with Ollama via LiteLLM (`dspy.LM('ollama_chat/qwen3:14b')`).

Your benchmark infrastructure (GT + F1 scoring) maps directly to DSPy's evaluation system.

- **Expected impact:** +0.02–0.05 F1 (manual prompt engineering has been extensive)
- **Risk:** Only 15 guidelines for optimization — small search space
- **Effort:** Medium (define signatures, run optimization — hours of GPU time)

---

## Phase 3: High Effort (1+ week each)

### 7. Vision Table Extraction Improvements (NI9RV3E7 — current F1=0.65)

NI9RV3E7 (217 GT recs) uses vector-rendered tables. Current best: mistral-small3.2:24b (F1=0.65, P=0.79, R=0.55).

#### 7A. Hybrid Structure+Vision Pipeline

- Use `img2table` or Microsoft Table Transformer for cell-level bounding box detection
- Apply OpenCV color preprocessing per cell (NI9RV3E7 uses color coding: green=I, yellow=IIa, orange=IIb, red=III)
- Per-cell OCR or vision extraction instead of full-page → less hallucination
- Color detection alone could determine grades without OCR

#### 7B. Table Page Classification

- Currently pdfplumber detects 35 "table pages" but many are non-recommendation tables
- Keyword heuristic or color-pattern detection could filter to ~15–20 relevant pages

#### 7C. Alternative Vision Models

| Model | Size | Why promising |
|-------|------|---------------|
| minicpm-v:8b | 4.9GB | MiniCPM-V 2.6: excellent table extraction in benchmarks |
| llava:13b | 7.4GB | Established vision-language model, less hallucination-prone |

### 8. Better PDF Text Extraction (Marker)

[Marker](https://github.com/datalab-to/marker) converts PDF to Markdown+JSON with high accuracy. Python 3.9 compatible (unlike Docling which requires 3.10+). Has `--use_llm` mode for highest accuracy. ~2GB VRAM.

Won't fix the precision problem (false positives are correctly extracted text) but could improve grade/level accuracy where it drops to 0.50–0.67 due to table parsing failures.

### 9. Fine-Tuning qwen3:14b

Fine-tune on 10–15 guideline examples to learn domain-specific recommendation boundaries. Highest potential impact but requires data preparation, LoRA setup, and careful validation to avoid overfitting on small dataset.

---

## Closed / Completed

| # | Improvement | Status | Outcome (validated on 15 datasets) |
|---|-------------|--------|-------------------------------------|
| — | Vision extraction (NI9RV3E7) | **DONE** | F1 0.00→0.65 with mistral-small3.2:24b |
| — | Grading scheme detection | **DONE** | Fixed 0% ERS grade accuracy |
| — | Cross-scheme few-shot fallback | **DONE** | BDYDTUHA F1 +0.06 |
| — | Parser hardening | **DONE** | Zero-cost defensive improvement |
| — | Token overlap verification | **DONE** | Near-zero impact (Δ=-0.013 on 15ds), kept as safety net |
| — | **Adaptive self-consistency** | **DONE** | **F1=0.783 (+0.076). New best config.** Cluster-adaptive thresholds fix small/large dataset failure modes. |
| — | Self-consistency voting (fixed) | **DONE** | F1=0.750 (+0.043). Superseded by adaptive version. |
| — | Post-extraction filter (qwen3:8b) | **CLOSED** | F1=0.699 (-0.008). Said YES to everything for 9/14 guidelines; 6× slower. |
| — | JSON schema enforcement (re-test) | **CLOSED** | F1=0.498 (-0.209). Still massively over-extracts even with schema (P=0.38). |
| — | CoT prompt | **CLOSED** | Causes regressions (model overthinks) |
| — | Improved few-shot selection | **CLOSED** | Both stratified and negative examples regress |
| — | JSON structured output | **CLOSED** | F1=0.47 (5ds). Re-tested with schema enforcement: F1=0.498 (15ds). |
| — | Two-pass extraction | **CLOSED** | llama3.2 classifier too conservative |
| — | Ensemble (union+dedup) | **CLOSED** | Too many false positives (avg F1=0.42) |
| — | 3-page chunking | **CLOSED** | Δ=-0.029 on 15ds, pathologically slow on ICU (2.8hr/guideline) |
| — | Alias expansion | **DONE** | Included in grading scheme fix |
| — | Logprob confidence filtering | **SKIP** | FPs are real text, not hallucinations — model is "confident" about wrong classifications |

## Key Insight: The Precision Gap

qwen3:14b's false positives are real text from the PDF (background statements, evidence summaries) misclassified as recommendations — not hallucinations. This is fundamentally a **classification problem**, not a grounding problem. Approaches that verify source text (token overlap, logprobs) don't help. The most promising fix is a dedicated post-extraction classifier (Phase 1, items 1A/1B).

---

## References

- Clinical ModernBERT: https://arxiv.org/abs/2504.03964
- NuExtract 2.0: https://numind.ai/blog/nuextract-2-0
- NuExtract on Ollama: https://ollama.com/library/nuextract
- DSPy framework: https://dspy.ai/
- Marker PDF extraction: https://github.com/datalab-to/marker
- img2table (cell detection): https://github.com/xavctn/img2table
- Microsoft Table Transformer: https://huggingface.co/microsoft/table-transformer-structure-recognition
- bespoke-minicheck (NLI verification): https://huggingface.co/bespoke-stratos/Bespoke-Minicheck-7B
- Ollama structured outputs: https://docs.ollama.com/capabilities/structured-outputs
