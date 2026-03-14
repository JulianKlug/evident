# A/B Test Results (15 datasets, 2026-03-14)

## Summary (excluding NI9RV3E7, 14 text guidelines)

| Test | Avg F1 | ΔF1 | Avg P | Avg R | Avg Grade | Avg Level |
|------|--------|-----|-------|-------|-----------|-----------|
| baseline | 0.707 | — | 0.608 | 0.936 | 0.881 | 0.940 |
| sc_adaptive | 0.783 | +0.076 | 0.788 | 0.841 | 0.892 | 0.966 |
| self_consistency | 0.750 | +0.043 | 0.820 | 0.766 | 0.891 | 0.923 |
| **ml_filter** | **0.732** | **+0.025** | **0.646** | **0.935** | **0.936** | — |
| no_normalize | 0.726 | +0.019 | 0.639 | 0.936 | 0.860 | 0.940 |
| post_filter | 0.699 | -0.008 | 0.630 | 0.883 | 0.901 | 0.957 |
| zero_shot | 0.697 | -0.010 | 0.600 | 0.936 | 0.784 | 0.908 |
| verify | 0.694 | -0.013 | 0.590 | 0.924 | 0.897 | 0.952 |
| grading_oracle | 0.676 | -0.031 | 0.552 | 0.949 | 0.874 | 0.839 |
| chunking_3page | 0.678 | -0.029 | 0.577 | 0.917 | 0.948 | 0.955 |
| auto_vision | 0.663 | -0.045 | 0.562 | 0.931 | 0.872 | 0.894 |
| json_schema | 0.498 | -0.209 | 0.378 | 0.939 | 0.903 | 0.937 |

Note: ml_filter results are from LOGO-CV (leave-one-guideline-out cross-validation), not a traditional A/B test run, since the classifier is trained on the same 15 guidelines. Each guideline's prediction comes from a model that never saw it.