# Best Model Configuration

Last updated: 2026-03-13 (validated on 15-dataset benchmark)

## Recommended Commands

```bash
# Standard (best simple config)
python run_benchmark.py --model qwen3:14b --few-shot-only --normalize

# With adaptive self-consistency (best F1, 3× slower)
python run_benchmark.py --model qwen3:14b --few-shot-only --normalize --self-consistency --n-samples 3 --sc-temperature 0.3 --adaptive-threshold

# With auto-vision for opaque-table PDFs (e.g., NI9RV3E7)
python run_benchmark.py --model qwen3:14b --few-shot-only --normalize --auto-vision --vision-model mistral-small3.2:24b
```

## Text Extraction

| Setting | Value |
|---------|-------|
| Model | qwen3:14b (9.3GB) |
| Strategy | few_shot |
| Normalization | enabled |
| Output format | pipe (default) |
| Pages per chunk | 1 (default) |

### Performance — 15-Dataset Benchmark (excluding NI9RV3E7)

| Config | Avg F1 | Avg P | Avg R | Avg Grade | Avg Level |
|--------|--------|-------|-------|-----------|-----------|
| **SC Adaptive** | **0.783** | 0.788 | 0.841 | **0.892** | — |
| SC Fixed (consensus=2) | 0.750 | **0.820** | 0.766 | 0.891 | — |
| Baseline | 0.707 | 0.608 | **0.936** | 0.881 | 0.940 |

### Per-Guideline Breakdown (Baseline)

| Guideline | Scheme | GT | Extr | F1 | P | R | Grade | Level |
|-----------|--------|-----|------|------|------|------|-------|-------|
| ACP:48AJE2AR | GRADE | 2 | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ACP:XK8ZAXYM | GRADE | 3 | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ACP:PAEHSPH3 | GRADE | 3 | 4 | 0.857 | 0.750 | 1.000 | 1.000 | 1.000 |
| ACP:XBAJSPZE | GRADE | 2 | 3 | 0.800 | 0.667 | 1.000 | 1.000 | 1.000 |
| ACP:XLMXNL32 | GRADE | 6 | 10 | 0.750 | 0.600 | 1.000 | 0.833 | 0.833 |
| ERS:BDYDTUHA | ABCD_123 | 60 | 43 | 0.699 | 0.837 | 0.600 | 0.972 | 0.861 |
| ERS:CMCZFLU4 | GRADE | 12 | 17 | 0.690 | 0.588 | 0.833 | 0.700 | 1.000 |
| ACP:WND8NBNA | GRADE | 5 | 10 | 0.667 | 0.500 | 1.000 | 1.000 | 0.800 |
| ACP:8J2P9MD8 | GRADE | 4 | 8 | 0.667 | 0.500 | 1.000 | 1.000 | 1.000 |
| ACP:89499SID | GRADE | 1 | 2 | 0.667 | 0.500 | 1.000 | 1.000 | 1.000 |
| ICU:10_1007_s00134-025-08058-x | GRADE | 3 | 6 | 0.667 | 0.500 | 1.000 | 0.667 | 1.000 |
| ICU:10_1007_s00134-024-07369-9 | GRADE | 11 | 27 | 0.579 | 0.407 | 1.000 | 1.000 | 1.000 |
| ACP:8V9WED94 | GRADE | 3 | 4 | 0.571 | 0.500 | 0.667 | 0.500 | 1.000 |
| ICU:10_1007_s00134-025-07840-1 | GRADE | 3 | 18 | 0.286 | 0.167 | 1.000 | 0.667 | 0.667 |
| ERS:NI9RV3E7 | ESC_ERS | 217 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Vision Extraction (for opaque-table PDFs)

| Setting | Value |
|---------|-------|
| Vision model | mistral-small3.2:24b (15GB) |
| Detection | auto (--auto-vision) |
| Generation params | temperature=0, top_p=0.1, repeat_penalty=1.1 |

### NI9RV3E7 Performance (auto-vision)

| Metric | Value |
|--------|-------|
| F1 | 0.65 |
| Precision | 0.79 |
| Recall | 0.55 |
| Grade Accuracy | 0.94 |
| Level Accuracy | 0.97 |

## Active Enhancements

| Enhancement | Impact (15ds) | Flag |
|-------------|---------------|------|
| Adaptive self-consistency (3 samples) | **+0.076 F1**, P 0.61→0.79, R 0.94→0.84 | `--self-consistency --adaptive-threshold` |
| Cross-scheme few-shot fallback | BDYDTUHA F1 +0.06 | automatic |
| Parser hardening | defensive, no regression | automatic |
| Token overlap verification | near-zero impact | `--verify` |

## Approaches Tested and Rejected (validated on 15 datasets)

| Approach | Avg F1 (15ds) | Why Rejected |
|----------|--------------|-------------|
| Post-extraction filter (qwen3:8b) | 0.699 (-0.008) | Said YES to everything for 9/14 guidelines; 6× slower |
| 3-page chunking | 0.678 (-0.029) | Hurts precision, pathologically slow on ICU |
| Auto-vision on all | 0.663 (-0.045) | False table detection hurts text guidelines |
| Stratified few-shot selection | 0.65 (5ds) | Specific examples matter more than diversity |
| Negative prompt examples | 0.63 (5ds) | Over-conservatism, kills recall |
| JSON schema enforcement | 0.498 (-0.209) | Still massively over-extracts even with schema (P=0.38) |
| JSON structured output | 0.47 (5ds) | Over-extracts (hallucinated recs) |
| Two-pass (llama3.2 classifier) | 0.47 (5ds) | Classifier too conservative, kills recall |
| Ensemble (union+dedup) | 0.42 (5ds) | Union adds too many false positives |
| CoT prompt | < baseline (5ds) | Model overthinks, causes regressions |
| gemma3:27b vision | 0.30 (NI9RV3E7) | Heavy hallucination (995 raw recs) |
| qwen2.5vl:7b vision | N/A | Crashes (GGML assertion error) |
