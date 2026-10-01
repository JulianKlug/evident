"""Tests for extraction.self_consistency: per-cluster voting on new columns."""

import pytest

from evaluation.grading import GRADE
from extraction.pdf_loader import PDFPage
from extraction.prompts import PromptStrategy
from extraction.self_consistency import self_consistency_extract
from tests.legacy_golden import FakeEncoder, MockClient

_REC = "Reassess volume status frequently in shock"


def _run_sc(monkeypatch, responses, n_pages=1, **kwargs):
    import extraction.self_consistency as sc_mod
    pages = [PDFPage(page_number=i + 1, text=f"page {i + 1}") for i in range(n_pages)]
    monkeypatch.setattr(sc_mod, "load_pdf_pages", lambda source: pages)
    client = kwargs.pop("client", None) or MockClient(responses)
    result = self_consistency_extract(
        "fake.pdf", strategy=PromptStrategy(name="zero_shot", scheme=GRADE), client=client,
        similarity_model=FakeEncoder(), n_samples=3, **kwargs,
    )
    return result.recommendations_df


def test_category_is_majority_vote(monkeypatch):
    df = _run_sc(monkeypatch, [
        f"{_REC} | BEST_PRACTICE | NA",
        f"{_REC} | BEST_PRACTICE | NA",
        f"{_REC} | Strong For | High",
    ])
    assert df["category"].tolist() == ["best_practice"]


def test_cluster_keeps_min_page_and_its_chunk(monkeypatch):
    # 2 pages per sample, rec found on page 2 by two samples and page 1 by one
    df = _run_sc(monkeypatch, [
        "NO_RECOMMENDATIONS_FOUND", f"{_REC} | Strong For | High",
        f"{_REC} | Strong For | High", "NO_RECOMMENDATIONS_FOUND",
        "NO_RECOMMENDATIONS_FOUND", f"{_REC} | Strong For | High",
    ], n_pages=2)

    assert df[["page", "chunk_id"]].values.tolist() == [[1, 0]]


class _SeedRecordingClient(MockClient):
    def __init__(self, responses):
        super().__init__(responses)
        self.seeds = []

    def generate(self, prompt, num_ctx=None, temperature=0, **kwargs):
        self.seeds.append(kwargs.get("seed", "unset"))
        return super().generate(prompt, num_ctx=num_ctx, temperature=temperature)


def test_sample_k_uses_seed_plus_k(monkeypatch):
    client = _SeedRecordingClient([f"{_REC} | Strong For | High"] * 3)
    _run_sc(monkeypatch, [], client=client, seed=10)
    assert client.seeds == [10, 11, 12]


def test_no_seed_kwarg_when_unseeded(monkeypatch):
    client = _SeedRecordingClient([f"{_REC} | Strong For | High"] * 3)
    _run_sc(monkeypatch, [], client=client)
    assert client.seeds == ["unset"] * 3
