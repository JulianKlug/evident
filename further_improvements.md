# Further Improvements for the Extraction Pipeline

Based on codebase analysis, literature review, and benchmark results. Organized by expected impact and effort.

---

## 1. Vision-Based Table Extraction (Critical — fixes NI9RV3E7)

**Problem:** NI9RV3E7 (217 GT recs) stores recommendations in vector-rendered PDF tables where cell text is drawn as paths/glyphs, not extractable text. ALL text-based extractors fail:

| Tool | Result |
|------|--------|
| PyPDF | Captures footnotes only, no table content |
| Docling (TableFormer) | Detects 35 tables, extracts 0 cell content |
| pdfplumber | Detects table structure but no text in cells |
| Tesseract OCR | Garbled output on colored table backgrounds |
| RapidOCR (PaddleOCR) | Cannot detect English text at all |

**Current approach:** Vision LLM (gemma3:27b) on full-page images rendered at 300 DPI.
- Implementation: `extraction/vision_extractor.py` + `load_pdf_table_images()` in `pdf_loader.py`
- Results: F1=0.27 (P=0.17, R=0.71), Grade Acc=0.50, Level Acc=0.42
- Major issue: ~80 hallucinated "The use of X is not recommended" entries

**See Section 12 for detailed improvement strategies.**

---

## 2. Self-Consistency Voting (High impact, medium effort)

**Problem:** LLM extraction is non-deterministic — borderline recommendations sometimes get extracted, sometimes not. A single run may miss valid recommendations or include spurious ones.

**Solution:** Run extraction N times (e.g., 3) per chunk with temperature > 0, then keep recommendations appearing in >= 2 runs.

**Implementation:**
- Add `n_samples` parameter to `extract_guideline()`
- For each chunk, generate N responses with temperature=0.3
- Use existing similarity-based dedup to cluster extracted recs across samples
- Keep recs appearing in majority of samples (consensus filtering)

**Expected impact:** +5-10% recall on borderline cases, reduced false positives

**References:**
- [Self-Consistency Prompting](https://www.promptingguide.ai/techniques/consistency)

---

## 3. Verification Pass (High impact, medium effort)

**Problem:** LLMs sometimes hallucinate recommendations or misattribute grades. No verification that extracted text actually exists in the source.

**Solution:** After extraction, add a verification step where the LLM (or a simpler check) confirms each recommendation exists verbatim in the source text.

**Implementation options:**
- **String matching:** Check if extracted text has high overlap with any passage in the source page (fuzzy matching with >80% token overlap)
- **LLM verification:** Present each extracted rec + source text to the LLM and ask "Does this recommendation appear in the text? YES/NO"
- **Dual-LLM adversarial:** One LLM extracts, another critiques (shown to work well in 2025 bioRxiv paper)

**Expected impact:** +5-15% precision (removes hallucinated recs)

**References:**
- [Dual-LLM Adversarial Framework for IE](https://www.biorxiv.org/content/10.1101/2025.09.11.675507v1.full)
- [Iterative Refinement for Clinical IE](https://www.nature.com/articles/s41746-025-01686-z)

---

## 4. Chain-of-Thought Extraction Prompt (Medium impact, low effort)

**Problem:** The current prompt asks the LLM to directly output recommendations. Models sometimes extract background statements or summaries that aren't actual graded recommendations.

**Solution:** Add a reasoning step to the prompt:

```
First, scan the text and identify all statements that:
1. Direct a clinical action (e.g., "should be considered", "is recommended", "should not be used")
2. Have an explicitly stated grade/class AND level of evidence

Then, for each identified statement, extract the exact text, grade, and level.
```

**Implementation:** Modify `build_prompt()` in `extraction/prompts.py` to include CoT instruction. Parse thinking from output before extracting recommendations.

**Expected impact:** +5-10% precision (fewer false positives from background text)

**References:**
- [Chain-of-Thought Prompting (IBM)](https://www.ibm.com/think/topics/chain-of-thoughts)

---

## 5. Improved Few-Shot Example Selection (Medium impact, low effort)

**Problem:** Few-shot examples are randomly sampled from ground truth. Some examples may be atypical or not representative of the recommendation format in the target guideline.

**Current code:** `extraction/datasets.py` line 145-146 uses random sampling.

**Improvements:**
- **Diverse selection:** Pick examples covering different grade/level combinations
- **Difficulty-aware:** Include one "easy" (clear-cut) and one "hard" (subtle) example
- **Format-matching:** For guidelines using tables, show table-extracted examples; for inline, show inline
- **Negative examples:** Add 1-2 examples of what NOT to extract (already partially done in prompts.py but could be more specific per scheme)

**Expected impact:** +3-5% F1 from better LLM calibration

---

## 6. Response Parser Hardening (Medium impact, low effort)

**Problem:** The response parser (`extraction/response_parser.py`) silently drops malformed output. If 4 out of 5 recommendations parse correctly but one has a formatting error, all may be discarded or the malformed one is lost.

**Specific issues:**
- Boilerplate detection (line 65) is fragile — triggers on pipes OR numbers
- Pipe-delimited parsing (line 104) uses a single regex that fails on pipes within recommendation text
- CSV fallback (line 136-139) doesn't handle quoted fields
- `NO_RECOMMENDATIONS_FOUND` sentinel requires exact match (line 38)

**Improvements:**
- Add partial recovery: parse each line independently, keep valid ones
- Add fuzzy sentinel detection: case-insensitive, allow variations
- Add logging of parse failures for debugging
- Handle common LLM preambles ("Based on the text...", "Here are...")

**Expected impact:** +2-5% recall (recovered from parse failures)

---

## 7. Grading Scheme Alias Expansion (Medium impact, low effort)

**Problem:** Grade/level normalization in `evaluation/grading.py` has incomplete alias coverage, causing false negatives in accuracy metrics.

**Missing aliases identified:**
- ABCD_123: "a"/"b"/"c"/"d" without "grade" prefix; "level 1"/"level 2" without "of evidence"
- GRADE: "strong recommendation" (without for/against); "conditional" → "Weak"
- ESC_ERS: "class 1"/"class 2a" (numeric variants); spacing variants "II a" vs "IIa"
- Typo: ABCD_123 has "iit" → "2" (should be "iii" → "3"?)

**Expected impact:** +5-15% grade/level accuracy (especially for ERS where accuracy is 0%)

---

## 8. Token-Aware Chunking (Medium impact, medium effort)

**Problem:** Current chunking uses fixed `pages_per_chunk` regardless of page length. Some pages are half-empty, others are dense. A 3-page chunk might be 500 tokens or 5000 tokens.

**Solution:** Chunk by token count rather than page count:
- Estimate tokens per page (rough: word_count * 1.3)
- Set target chunk size based on model context window (e.g., 70% of available context after prompt)
- Merge short pages, split long ones

**Expected impact:** Better context utilization, especially for large guidelines

---

## 9. NuExtract as Alternative Extraction Model (Medium impact, high effort)

**Problem:** General-purpose LLMs (qwen, deepseek) are not optimized for structured extraction.

**Solution:** [NuExtract](https://numind.ai/blog/nuextract-2-0) is a fine-tuned extraction model (0.5B-7B params) that outperforms GPT-4o on structured extraction benchmarks. It's available on HuggingFace and could run via Ollama.

**Considerations:**
- Purpose-built for structured extraction from text
- Much smaller and faster than current models
- May need domain-specific fine-tuning for medical terminology
- Could serve as a fast first-pass extractor

**Expected impact:** Potentially significant, but needs evaluation

---

## 10. Ensemble / Model Selection per Guideline Type (Medium impact, medium effort)

**Problem:** No single model dominates across all guidelines. qwen3:14b is best for ACP (GRADE scheme), deepseek-r1:32b is best for ERS (ABCD_123 scheme).

**Solution:** Use guideline metadata to select the best model:
- If GRADE scheme → use qwen3:14b
- If ABCD_123/ESC_ERS scheme → use deepseek-r1:32b
- Or: run multiple models and merge results (ensemble)

**Ensemble approach:**
- Run 2 models on same input
- Merge results using similarity-based dedup (keep unique recs from both)
- Weight by model confidence or past performance on the scheme

**Expected impact:** +10-15% F1 by using best model per scenario

---

## 11. Adaptive Similarity Thresholds (Low-medium impact, low effort)

**Problem:** Fixed similarity threshold (0.65 in benchmark, 0.95 default in matching) is suboptimal. Short recommendations may need lower thresholds; long ones higher.

**Solution:**
- Compute optimal threshold per dataset from validation data
- Or use length-adaptive thresholds: shorter text → lower threshold
- Standardize thresholds across codebase (currently inconsistent)

**Expected impact:** +2-5% in matching accuracy

---

## 12. Vision Table Extraction Improvements (P0 — active development)

Vision extraction is now the primary approach for NI9RV3E7 (and any PDF with vector-rendered tables). Current F1=0.27. Three categories of improvement:

### 12A. Hallucination Mitigation (Quick wins)

**Problem:** gemma3:27b fabricates ~80 fake recommendations ("The use of cloud computing is not recommended", "The use of CRISPR-Cas9 is not recommended", etc.). These are class III/C entries that don't exist in the source.

**Strategies (ordered by expected impact):**

1. **Tighter generation parameters** — Set `top_p=0.1`, `repeat_penalty=1.0` in addition to `temperature=0`. Reduces creative/repetitive generation.

2. **Anti-hallucination prompt instructions** — Add explicit rules:
   - "Only extract rows that are VISIBLE in the table image"
   - "Do NOT generate, infer, or extrapolate recommendations"
   - "If a row is partially visible, skip it"
   - "Count the number of visible table rows and ensure your output has the same count"

3. **Post-hoc verification with bespoke-minicheck** — A small NLI model that checks if each extracted claim is grounded in the source. Run on page image OCR text vs extracted recs. Remove ungrounded claims.
   - Package: `bespoke-minicheck` (available on HuggingFace, ~400MB)
   - Expected: eliminates most hallucinated entries

4. **Two-pass OCR+Vision consensus** — Run both OCR and vision extraction, keep only recommendations that appear in BOTH outputs. Even if OCR is garbled, enough signal may overlap.

5. **JSON structured output** — Use Ollama's `format` parameter with JSON schema to constrain output structure. Prevents freeform generation of fake entries.

### 12B. Better Vision Models (Medium effort)

**Problem:** gemma3:27b hallucinates heavily. Alternative models may be more faithful.

| Model | Size | Why promising |
|-------|------|---------------|
| **qwen2.5-vl:7b** | 4.7GB | Purpose-built for visual document understanding; strong OCR capability |
| **minicpm-v:8b** | 4.9GB | MiniCPM-V 2.6: excellent table extraction in benchmarks |
| **llava:13b** | 7.4GB | Established vision-language model, less hallucination-prone |

**Recommendation:** Try `qwen2.5-vl:7b` first — it's specifically trained for document/table understanding and is small enough to run alongside other models.

### 12C. Hybrid Structure+Vision Pipeline (Higher effort, highest potential)

**Problem:** Vision LLMs process the whole page at once, making it hard to extract individual cells accurately. Table structure detection + per-cell extraction would be more precise.

**Approach 1: img2table + Color Preprocessing + OCR**
- Use `img2table` library for cell-level bounding box detection (no ML required, uses line detection)
- Apply OpenCV color preprocessing per cell:
  ```python
  # Convert colored cells to high-contrast B&W
  hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
  # Detect cell background color (green=I, orange=IIa, etc.)
  # Invert/threshold to make text readable for OCR
  ```
- Run OCR (Tesseract or PaddleOCR) on preprocessed cells
- Map cell positions to columns (Recommendation, Class, Level)
- **Key insight:** NI9RV3E7 uses color coding — cell background color directly maps to recommendation class (green=I, yellow=IIa, orange=IIb, red=III). Color detection alone could determine grades without OCR.

**Approach 2: Microsoft Table Transformer (TATR)**
- Pre-trained model for table structure recognition (cell-level detection)
- Available via HuggingFace: `microsoft/table-transformer-structure-recognition`
- Provides row/column/cell bounding boxes → crop individual cells → OCR or vision per cell
- More robust than line-based detection for complex table layouts

**Approach 3: Cell-Level Vision Extraction**
- Detect table structure (img2table or TATR)
- For each cell, crop and send to vision LLM individually
- Much smaller context = less hallucination risk
- Slower but more accurate per-cell

### 12D. Table Page Classification (Low effort, reduces noise)

**Problem:** pdfplumber detects 35 "table pages" but many contain definition tables, classification tables, or prognostic factor tables — not recommendation tables.

**Strategies:**
1. **Keyword heuristic:** Check if page text/OCR contains "recommendation", "class", "level of evidence"
2. **Color-based:** NI9RV3E7 recommendation tables have colored cells (green/yellow/orange/red). Detect pages with these color patterns.
3. **LLM pre-classification:** Quick vision pass asking "Does this page contain a recommendation table? YES/NO" before full extraction

**Expected impact of all Section 12 improvements combined:** NI9RV3E7 F1 from 0.27 → 0.55-0.70

---

## Priority Matrix

| # | Improvement | Impact | Effort | Priority |
|---|-------------|--------|--------|----------|
| 12A | Vision hallucination mitigation | Critical | Low-Med | **P0** |
| 12B | Try qwen2.5-vl:7b vision model | High | Low | **P0** |
| 12D | Table page classification | Medium | Low | **P0** |
| 7 | Alias expansion | Medium-High | Low | **P1** |
| 12C | Hybrid structure+vision pipeline | High | High | **P1** |
| 6 | Parser hardening | Medium | Low | **P1** |
| 4 | CoT prompt | Medium | Low | **P1** |
| 10 | Model selection/ensemble | Medium-High | Medium | **P1** |
| 3 | Verification pass (bespoke-minicheck) | High | Medium | **P2** |
| 2 | Self-consistency voting | High | Medium | **P2** |
| 5 | Few-shot selection | Medium | Low | **P2** |
| 8 | Token-aware chunking | Medium | Medium | **P2** |
| 11 | Adaptive thresholds | Low-Medium | Low | **P3** |
| 9 | NuExtract model | Medium | High | **P3** |

---

## References

- Docling: https://arxiv.org/html/2408.09869v1
- NuExtract 2.0: https://numind.ai/blog/nuextract-2-0
- Self-Consistency: https://www.promptingguide.ai/techniques/consistency
- Dual-LLM Adversarial IE: https://www.biorxiv.org/content/10.1101/2025.09.11.675507v1.full
- Iterative Clinical IE: https://www.nature.com/articles/s41746-025-01686-z
- PDF Parsing Benchmark: https://arxiv.org/html/2410.09871v1
- LLMs for Guideline Extraction (JMIR 2025): https://www.jmir.org/2025/1/e73486/PDF
- Table Extraction Benchmark (ACL 2025): https://aclanthology.org/2025.xllm-1.2/
- img2table (cell detection): https://github.com/xavctn/img2table
- Microsoft Table Transformer: https://huggingface.co/microsoft/table-transformer-structure-recognition
- bespoke-minicheck (NLI verification): https://huggingface.co/bespoke-stratos/Bespoke-Minicheck-7B
- Qwen2.5-VL: https://ollama.com/library/qwen2.5-vl
- MiniCPM-V: https://ollama.com/library/minicpm-v
