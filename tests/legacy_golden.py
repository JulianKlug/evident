"""Replay helpers for the legacy golden regression harness.

Shared by tests/test_legacy_golden.py (replay) and
tests/fixtures/legacy_golden/record.py (one-off recording).

    fixture JSON (pages + canned LLM responses)
            │
            ▼
    MockClient ──▶ extract_guideline / self_consistency_extract
            │
            ▼
    recommendations_df[LEGACY_COLUMNS] == expected_<case>.csv
"""

from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

from evaluation.grading import GRADE
from extraction.extractor import extract_guideline
from extraction.llm_client import LLMResponse
from extraction.pdf_loader import PDFPage
from extraction.prompts import PromptStrategy
from extraction.self_consistency import self_consistency_extract

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "legacy_golden")
LEGACY_COLUMNS = ["recommendation", "class", "LOE"]

_EMBEDDING_DIM = 256
_SC_SAMPLES = 3
_SC_TEMPERATURE = 0.3

# Golden runs: (run name, fixture name, runner, normalize)
RUNS = [
    ("acp_fewshot", "acp", "extract", False),
    ("acp_fewshot_norm", "acp", "extract", True),
    ("icu_fewshot", "icu", "extract", False),
    ("icu_fewshot_norm", "icu", "extract", True),
    ("synthetic_fewshot", "synthetic", "extract", False),
    ("synthetic_fewshot_norm", "synthetic", "extract", True),
    ("acp_sc", "acp", "sc", False),
    ("acp_sc_norm", "acp", "sc", True),
]


class MockClient:
    """Returns canned responses in call order. Signature matches the legacy call
    sites exactly, so any new kwarg (e.g. seed) leaking into a legacy run fails."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._call_count = 0
        self.model = "mock-model"
        self.model_info = {"context_window": 4096, "has_thinking": False}
        self.prompts: list[str] = []

    @property
    def has_thinking(self) -> bool:
        return False

    def generate(self, prompt, num_ctx=None, temperature=0):
        self.prompts.append(prompt)
        text = self._responses[self._call_count]
        self._call_count += 1
        return LLMResponse(raw_text=text, model=self.model, prompt_tokens=1,
                           eval_tokens=1, total_duration_ms=1.0)


class FakeEncoder:
    """Deterministic bag-of-words embedding (md5-hashed buckets, L2-normalized)."""

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), _EMBEDDING_DIM))
        for i, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % _EMBEDDING_DIM
                out[i, bucket] += 1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


def load_fixture(name: str) -> dict:
    with open(os.path.join(GOLDEN_DIR, f"{name}.json")) as f:
        return json.load(f)


def expected_path(run_name: str) -> str:
    return os.path.join(GOLDEN_DIR, f"expected_{run_name}.csv")


def run_case(run_name: str, monkeypatch) -> pd.DataFrame:
    """Replay one golden run through the legacy extractor with mocked I/O."""
    import extraction.extractor as extractor_mod
    import extraction.self_consistency as sc_mod

    _, fixture_name, runner, normalize = next(r for r in RUNS if r[0] == run_name)
    fx = load_fixture(fixture_name)

    # Replace PDF loading with the fixture's page texts
    pages = [PDFPage(page_number=i + 1, text=t) for i, t in enumerate(fx["pages"])]
    monkeypatch.setattr(extractor_mod, "load_pdf_pages", lambda source: pages)
    monkeypatch.setattr(sc_mod, "load_pdf_pages", lambda source: pages)

    strategy = PromptStrategy(name="few_shot", scheme=GRADE, examples=fx["examples"])

    if runner == "extract":
        client = MockClient(fx["responses"])
        result = extract_guideline("fake.pdf", strategy=strategy, client=client,
                                   pages_per_chunk=fx["pages_per_chunk"], normalize=normalize)
        return result.recommendations_df

    # SC: samples are consumed sample-major, chunk-minor
    flat = [r for sample in fx["sc_responses"] for r in sample]
    client = MockClient(flat)
    result = self_consistency_extract(
        "fake.pdf", strategy=strategy, client=client, similarity_model=FakeEncoder(),
        n_samples=_SC_SAMPLES, temperature=_SC_TEMPERATURE,
        pages_per_chunk=fx["pages_per_chunk"], normalize=normalize,
    )
    return result.recommendations_df
