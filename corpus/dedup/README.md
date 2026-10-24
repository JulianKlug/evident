# Within-guideline duplicate verdicts (Paper 1, spec §5)

- `paper1_verdicts.csv`: frozen LLM verdict for every screened pair of snapshot 8 (`scripts/paper1_dedup_adjudicate.py`). Hashed in `configs/paper1.json`; `publish --package` reads it.
- `audit/prompt_dev_*`: first screen (BioLORD/lexical ≥ 0.85, 338 pairs), first-prompt verdicts and an independent reviewer's labels; used to revise the prompt.
- `audit/blind_*`: 150 pairs outside the first screen (all 37 LLM duplicates + 113 random distinct), labelled blind to the verdicts (`blind_key.csv` holds them). Reviewer: a separate model instance; a human audit is pending.
