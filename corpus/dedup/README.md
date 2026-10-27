# Within-guideline duplicate verdicts (M4 draft, spec §5)

- `paper1_verdicts.csv`: frozen verdict for every screened pair of snapshot 8: duplicate only when both judges agree (`scripts/paper1_dedup_consensus.py`, spec §5 item 6). Hashed in `configs/paper1.json`; `publish --package` reads it.
- `judges/`: each judge's verdicts (`scripts/paper1_dedup_adjudicate.py`): qwen38-64k and gemma3-27b on all 1256 pairs; they split on 116 (`disagreements.csv`: 115 gemma-only duplicates, 1 qwen-only), which stay distinct: 177 duplicate pairs. mistral-small3.2-24b judged the 116 under the superseded majority rule (item 5) and is kept for audit only.
- `audit/human_*`: 15-pair stratified human check (`scripts/paper1_dedup_human_sample.py`).
- `audit/prompt_dev_*`: first screen (BioLORD/lexical ≥ 0.85, 338 pairs), first-prompt verdicts and an independent reviewer's labels; used to revise the prompt.
- `audit/blind_*`: 150 pairs outside the first screen (all 37 LLM duplicates + 113 random distinct), labelled blind to the verdicts (`blind_key.csv` holds them). Reviewer: a separate model instance; superseded by the 15-pair human check.

## Live map (M5)

- `live_verdicts.csv`: the verdict ledger live `publish` reads; seeded from `paper1_verdicts.csv`, then only grows. A verdict applies only while doi, ordinals and both texts match the screened pair; verdicts of replaced texts stay as history. New pairs: `scripts/paper1_dedup_adjudicate.py --snapshot ID --ledger corpus/dedup/live_verdicts.csv` per judge, then `scripts/paper1_dedup_consensus.py`, which merges into the ledger. `paper1_verdicts.csv` is never written again.
