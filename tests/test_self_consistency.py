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
