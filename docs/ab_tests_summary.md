# A/B Test Results (15 datasets, 2026-03-15)

## Summary (excluding NI9RV3E7, 14 text guidelines)

| Test | Avg F1 | ΔF1 | Avg P | Avg R | Avg Grade | Avg Level |
|------|--------|-----|-------|-------|-----------|-----------|
| baseline | 0.707 | — | 0.608 | 0.936 | 0.881 | 0.940 |
| **sc_adaptive_ml_filter** | **0.860** | **+0.153** | **0.937** | **0.818** | **0.861** | — |
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
| context_oracle | 0.690 | -0.017 | 0.571 | 0.949 | 0.887 | 0.919 |
| sc_adaptive_ml_filter_sc | 0.762 | -0.098* | 0.773 | 0.837 | 0.818 | — |
| sc_adaptive_context_oracle | 0.773 | -0.010* | 0.836 | 0.783 | 0.877 | 0.899 |

*SC Adaptive + Context Oracle delta is vs SC Adaptive (0.783), not baseline.

Note: ml_filter results are from LOGO-CV (leave-one-guideline-out cross-validation), not a traditional A/B test run, since the classifier is trained on the same 15 guidelines. Each guideline's prediction comes from a model that never saw it.

Note: sc_adaptive_ml_filter uses the baseline-trained ML classifier on SC Adaptive outputs. SC is stochastic (temp=0.3), so some per-guideline variation vs sc_adaptive is from sampling. ML filter removed: PAEHSPH3 1/4, CMCZFLU4 2/15, ICU-07369 5/12, ICU-07840 3/7, ICU-08058 2/5.

*sc_adaptive_ml_filter_sc delta is vs sc_adaptive_ml_filter (0.860). Used a classifier retrained on SC outputs (23 FPs) — too few examples, removed only 1 rec total. Baseline-trained classifier is superior.