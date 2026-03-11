# Best Model Configuration

Last updated: 2026-03-11

## Recommended Command

```bash
# Text-only guidelines (ACP, most ERS)
python run_benchmark.py --model qwen3:14b --few-shot-only --normalize

# Guidelines with opaque tables (e.g., NI9RV3E7)
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

### Performance (excluding NI9RV3E7)

| Metric | Value |
|--------|-------|
| Avg F1 | 0.83 |
| Avg Grade Accuracy | 0.84 |
| Avg Level Accuracy | 0.96 |

### Per-Guideline Breakdown

| Guideline | Scheme | F1 | P | R | Grade | Level |
|-----------|--------|------|------|------|-------|-------|
| ACP:XLMXNL32 | GRADE | 0.92 | 0.86 | 1.00 | 1.00 | 1.00 |
| ACP:8J2P9MD8 | GRADE | 0.89 | 0.80 | 1.00 | 1.00 | 1.00 |
| ACP:8V9WED94 | GRADE | 0.80 | 1.00 | 0.67 | 0.50 | 1.00 |
| ERS:CMCZFLU4 | GRADE | 0.87 | 0.77 | 1.00 | 0.80 | 1.00 |
| ERS:BDYDTUHA | ABCD_123 | 0.74 | 0.81 | 0.68 | 0.98 | 0.80 |

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

## Active Enhancements (zero-cost, kept in pipeline)

| Enhancement | Impact | Flag |
|-------------|--------|------|
| Cross-scheme few-shot fallback | BDYDTUHA F1 +0.06 | automatic |
| Parser hardening | defensive, no regression | automatic |
| Token overlap verification | safety net (+0.001 P) | `--verify` |

## Approaches Tested and Rejected

| Approach | Avg F1 | Why Rejected |
|----------|--------|-------------|
| Stratified few-shot selection | 0.65 | Specific examples matter more than diversity |
| Negative prompt examples | 0.63 | Over-conservatism, kills recall |
| Self-consistency voting (3 runs) | 0.78 | Runs too similar, consensus hurts BDYDTUHA |
| JSON structured output | 0.47 | Over-extracts (hallucinated recs) |
| Two-pass (llama3.2 classifier) | 0.47 | Classifier too conservative, kills recall |
| Ensemble (qwen3:14b + deepseek-r1:32b) | 0.42 | Union adds too many false positives |
| 3-page chunking | < baseline | Hurt ACP precision |
| CoT prompt | < baseline | Model overthinks, causes regressions |
| Token overlap verification | 0.83 (+0.001) | FPs are real text, not hallucinations |
| gemma3:27b vision | 0.30 (NI9RV3E7) | Heavy hallucination (995 raw recs) |
| qwen2.5vl:7b vision | N/A | Crashes (GGML assertion error) |
