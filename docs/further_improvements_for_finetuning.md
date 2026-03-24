# Fine-Tuning v4: Improvement Strategies

## Current State (v3, F1=0.808)

FT v3 used 551 examples (157 real + 394 synthetic via qwen3:14b self-distillation) with QLoRA on qwen3:14b. ACP guidelines are near-perfect (5/9 at F1=1.0), but ERS/ICU remain weak.

### Failure Analysis

| Guideline | Scheme | FT F1 | Best F1 (SC+ML) | Gap | Root Cause |
|-----------|--------|-------|-----------------|-----|------------|
| CMCZFLU4 | GRADE | 0.452 | 0.800 | -0.348 | Low P (0.37) + low R (0.58), only 17 real examples |
| BDYDTUHA | ABCD_123 | 0.425 | 0.604 | -0.179 | Very low R (0.28), real pages have 1-11 recs but synthetic caps at 3 |
| ICU:07840-1 | GRADE | 0.375 | 0.857 | -0.482 | Over-extraction (P=0.23), only 3 real examples |

## Improvement Strategies (Ranked by Expected Impact)

### 1. Better Synthetic Generation Model (HIGH IMPACT)
**Problem**: Self-distillation (qwen3:14b generating its own training data) creates error feedback loops where the model's biases get reinforced.

**Solution**: Use deepseek-r1:32b as the synthetic data generator. As a reasoning model, it produces higher-quality, more carefully structured guideline pages with accurate grade annotations.

**Implementation**: Add `--model` CLI arg to `generate_synthetic_data.py`, increase context window for larger models.

**Expected impact**: Better grade accuracy in synthetic data -> fewer mislabeled training examples -> higher precision on ERS/ICU.

### 2. Fix Density Mismatch (HIGH IMPACT)
**Problem**: BDYDTUHA has 1-11 recs/page (mean ~4.6) but synthetic `--max-recs-per-page` caps at 3. The model learns to stop extracting after 3 recs, causing recall collapse (R=0.28).

**Solution**: Scheme-specific weighted density distributions that match real-world rec density per page. ABCD_123 gets distributions extending to 8 recs/page.

**Expected impact**: BDYDTUHA recall should improve significantly (0.28 -> 0.50+).

### 3. Positive Example Emphasis via Duplication (MEDIUM IMPACT)
**Problem**: ~50/50 positive/negative split causes the model to be too conservative (low recall). SFTTrainer doesn't support sample weights.

**Solution**: Duplicate positive examples N times to shift balance (e.g., 2x -> ~70/30 positive-heavy). This acts as a loss weighting proxy.

**Expected impact**: Higher recall across all schemes, especially ICU where over-extraction (low P) suggests model learns confusing patterns from negatives.

### 4. Higher LoRA Rank + More Epochs (MEDIUM IMPACT)
**Problem**: v3 used r=8, 3 epochs. With ~1000 examples, the model may underfit.

**Solution**: r=16, alpha=32, 5 epochs, lr=1e-4 (halved), warmup=0.15, grad_accum=8.

**Expected impact**: Better convergence on harder schemes without overfitting ACP.

### 5. Curriculum Learning / Scheme Balancing (LOW IMPACT, FUTURE)
**Problem**: 9 ACP guidelines dominate training data. Model overfits to GRADE scheme.

**Solution**: Oversample ERS/ICU examples or train with scheme-balanced batches.

**Status**: Partially addressed by synthetic generation targeting underrepresented schemes.

### 6. FT + SC + ML Pipeline (LOW IMPACT, CLOSED)
**Problem**: Combining FT with SC+ML was tested in v3 and performed worse (F1=0.581).

**Root cause**: FT model is deterministic at temp=0; SC diversity requires stochastic sampling which degrades FT model quality. ML filter removes some correct FT predictions.

**Status**: Closed. Zero-shot FT is optimal for the fine-tuned model.

## v4 Training Plan

### Data Composition (~1000 examples)
| Component | Count |
|-----------|-------|
| Real page-level | 157 |
| Synthetic (deepseek-r1:32b, density-fixed) | ~500 |
| Positive duplicates (2x) | ~350 |
| **Total** | ~1000 |

### Hyperparameters
- LoRA: r=16, alpha=32, dropout=0.1
- Training: 5 epochs, lr=1e-4, warmup=0.15, grad_accum=8, batch=1
- Sequence: max_seq_length=3072 (reduce from 4096 to fit more grad_accum)

### Success Criteria
- Overall F1 > 0.860 (surpass SC+ML)
- CMCZFLU4 F1 > 0.60
- BDYDTUHA F1 > 0.50
- ICU:07840-1 F1 > 0.50
