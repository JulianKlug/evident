# Further Improvements for the Extraction Pipeline

Based on codebase analysis, literature review, and benchmark results. Organized by expected impact and effort.

---

## 1. PDF Table Extraction (Critical — fixes NI9RV3E7)

**Problem:** The 80-page ERS pulmonary hypertension guideline (NI9RV3E7, 217 GT recs) stores all recommendations in structured PDF tables. PyPDF's `extract_text()` captures surrounding text and footnotes but not the table cell content. This accounts for ~72% of all ground truth recommendations (217/302) and is the single largest source of recall loss.

**Solution: Replace pypdf with Docling for table-aware parsing**

| Library | Table Accuracy | Notes |
|---------|---------------|-------|
| **Docling** (IBM) | 93.6-97.9% | TableFormer deep learning model; best accuracy by far |
| pdfplumber | Good | Handles complex layouts; character-level detail |
| Camelot | 73% | Good for lattice (bordered) tables |
| Tabula-py | 67.9% | Simple tables only |
| Marker-pdf | Weak on tables | Good for general PDF→markdown but tables suffer |

**Implementation approach:**
- Add `docling` as optional dependency
- Create `extraction/table_extractor.py` that detects table pages and extracts structured content
- Hybrid strategy: use Docling for table-heavy pages, fall back to pypdf for text-only pages
- Extract table rows directly as structured recommendations (text, grade, level) without LLM

**Expected impact:** NI9RV3E7 F1 from 0.00 → 0.50+ (conservative estimate)

**References:**
- [Docling Technical Report](https://arxiv.org/html/2408.09869v1)
- [PDF Parsing Benchmark (arXiv)](https://arxiv.org/html/2410.09871v1)

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

## 12. Vision-Based Extraction for Scanned/Complex PDFs (Low priority)

**Problem:** Some PDF pages may be scanned images or have complex visual layouts.

**Findings from research:**
- Vision LLMs (GPT-4o, Gemini) are better at extracting text from table cells than OCR
- BUT numerical accuracy is poor — grade values (I, IIa, IIb) could be misread
- Practitioners recommend: "Don't use LLMs as OCR" — use good parser + LLM reasoning

**Recommendation:** Use Docling (item 1) for structure, LLM for semantics. Only fall back to vision-based if text extraction completely fails.

---

## Priority Matrix

| # | Improvement | Impact | Effort | Priority |
|---|-------------|--------|--------|----------|
| 1 | Docling table extraction | Critical | Medium | **P0** |
| 7 | Alias expansion | Medium-High | Low | **P1** |
| 6 | Parser hardening | Medium | Low | **P1** |
| 4 | CoT prompt | Medium | Low | **P1** |
| 10 | Model selection/ensemble | Medium-High | Medium | **P1** |
| 2 | Self-consistency voting | High | Medium | **P2** |
| 3 | Verification pass | High | Medium | **P2** |
| 5 | Few-shot selection | Medium | Low | **P2** |
| 8 | Token-aware chunking | Medium | Medium | **P2** |
| 11 | Adaptive thresholds | Low-Medium | Low | **P3** |
| 9 | NuExtract model | Medium | High | **P3** |
| 12 | Vision-based fallback | Low | High | **P3** |

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
