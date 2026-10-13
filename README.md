# evident

Living evidence map of clinical guidelines: every recommendation, its strength and
its certainty, extracted from guideline PDFs by a local LLM, stored in SQLite and
published as tables, figures and a static dashboard.

Design: `docs/designs/living-evidence-map.md`. Plan: `docs/roadmap.md`. Specs: `docs/specs/`.

## Setup

Install [uv](https://docs.astral.sh/uv/), then from the repo root:

| Machine | Command | Gets |
|---------|---------|------|
| any (e.g. a laptop fetching PDFs) | nothing: `uv run` creates the environment | corpus tools |
| GPU machine | `uv sync --extra pipeline --group dev` | pipeline + tests |

Python 3.9 and every version are pinned (`pyproject.toml`, `uv.lock`); uv installs them.

## Corpus tools (any machine)

```
uv run scripts/search_guidelines.py                       # PubMed → corpus/candidates.csv
uv run scripts/fetch_pdfs.py --email YOU@ORG --pdf-dir ./pdfs
uv run scripts/import_pdfs.py --pdf-dir ./pdfs missing --open   # macOS: missing links as Chrome tabs
uv run scripts/import_pdfs.py --pdf-dir ./pdfs import ~/Downloads
```

`import` files each downloaded PDF under its manifest name, matched by the DOI inside it.

## Pipeline (GPU machine)

```
uv run python -m evident snapshot --config configs/v0_qwen38-64k_dedup097.json
uv run python -m evident status
uv run python -m evident validate
uv run python -m evident publish            # out/tables, out/figures, out/dashboard
uv run python -m evident add PDF            # one new guideline, then re-publish
```

Outside the repo (not installed by uv):

| What | Where | Checked by |
|------|-------|------------|
| Ollama with `qwen38-64k:latest` (digest `12af24e3ee72`) | `ollama list` | version id; drift → `ArtifactMismatchError` |
| BioLORD-2023 (`FremyCompany/BioLORD-2023`, revision `167aab52`) | Hugging Face cache | version id |
| Guideline PDFs | `/mnt/data1/klug/datasets/evidence_extraction/pdfs` | `status` coverage |
| Ground truth (ACP, ICU) | `/mnt/data1/klug/datasets/evidence_extraction` | `validate` (12 labelled guidelines) |

## Tests

```
uv run pytest tests -m "not slow"
```
