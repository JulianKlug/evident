# Within-guideline duplicate verdicts (Paper 1, spec §5)

- `paper1_verdicts.csv`: frozen consensus verdict for every screened pair of snapshot 8 (`scripts/paper1_dedup_consensus.py`). Hashed in `configs/paper1.json`; `publish --package` reads it.
- `judges/`: each judge's verdicts (`scripts/paper1_dedup_adjudicate.py`): qwen38-64k and gemma3-27b on all 1256 pairs; mistral-small3.2-24b on the 116 they split (`disagreements.csv`: 115 gemma-only duplicates, 1 qwen-only). Consensus: 198 duplicate pairs (qwen alone: 178).
- `audit/human_*`: 15-pair stratified human check (`scripts/paper1_dedup_human_sample.py`).
- `audit/prompt_dev_*`: first screen (BioLORD/lexical ≥ 0.85, 338 pairs), first-prompt verdicts and an independent reviewer's labels; used to revise the prompt.
- `audit/blind_*`: 150 pairs outside the first screen (all 37 LLM duplicates + 113 random distinct), labelled blind to the verdicts (`blind_key.csv` holds them). Reviewer: a separate model instance; superseded by the 15-pair human check.
