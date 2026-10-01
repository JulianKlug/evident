"""Tests for extraction.extractor with mock LLM client."""

import pandas as pd
import pytest

from extraction.extractor import extract_guideline, ExtractionResult
from extraction.llm_client import LLMResponse, OllamaClient
from extraction.prompts import PromptStrategy
from evaluation.grading import GRADE


class MockOllamaClient:
    """Mock client that returns canned responses."""

    def __init__(self, responses: list):
        self._responses = responses
        self._call_count = 0
        self.model = "mock-model"
        self.model_info = {"context_window": 4096, "has_thinking": False}

    @property
    def has_thinking(self):
        return False

    def generate(self, prompt, num_ctx=None):
        if self._call_count < len(self._responses):
            text = self._responses[self._call_count]
        else:
            text = "NO_RECOMMENDATIONS_FOUND"
        self._call_count += 1
        return LLMResponse(
            raw_text=text,
            model="mock",
            prompt_tokens=100,
            eval_tokens=50,
            total_duration_ms=10.0,
        )


class MockPDFPages:
    """Monkey-patch load_pdf_pages for testing."""
    pass


class TestExtractGuideline:
    def test_basic_extraction(self, monkeypatch):
        """Test pipeline with mock LLM returning pipe-delimited results."""
        from extraction.pdf_loader import PDFPage
        import extraction.extractor as extractor_mod

        mock_pages = [
            PDFPage(page_number=1, text="Page 1 text"),
            PDFPage(page_number=2, text="Page 2 text"),
        ]
        monkeypatch.setattr(extractor_mod, "load_pdf_pages", lambda source: mock_pages)

        mock_client = MockOllamaClient([
            "Use beta-blockers for HF | Strong For | High",
            "Use ACE inhibitors for HTN | Weak For | Moderate",
        ])

        strategy = PromptStrategy(name="zero_shot", scheme=GRADE)
        result = extract_guideline("fake.pdf", strategy=strategy, client=mock_client)

        assert isinstance(result, ExtractionResult)
        assert result.n_pages == 2
        assert result.n_pages_with_recs == 2
        assert result.n_raw_recommendations == 2
        assert result.n_final_recommendations == 2
        assert len(result.recommendations_df) == 2
        assert list(result.recommendations_df.columns) == ["recommendation", "class", "LOE", "category", "page", "chunk_id"]

    def test_dedup_across_pages(self, monkeypatch):
        """Same recommendation on two pages should be deduplicated."""
        from extraction.pdf_loader import PDFPage
        import extraction.extractor as extractor_mod

        mock_pages = [
            PDFPage(page_number=1, text="Page 1"),
            PDFPage(page_number=2, text="Page 2"),
        ]
        monkeypatch.setattr(extractor_mod, "load_pdf_pages", lambda source: mock_pages)

        mock_client = MockOllamaClient([
            "Use beta-blockers for HF | Strong For | High",
            "Use beta-blockers for HF | Strong For | High",  # Duplicate
        ])

        strategy = PromptStrategy(name="zero_shot", scheme=GRADE)
        result = extract_guideline("fake.pdf", strategy=strategy, client=mock_client)

        assert result.n_raw_recommendations == 2
        assert result.n_final_recommendations == 1

    def test_no_recommendations(self, monkeypatch):
        """All pages return NO_RECOMMENDATIONS_FOUND."""
        from extraction.pdf_loader import PDFPage
        import extraction.extractor as extractor_mod

        mock_pages = [PDFPage(page_number=1, text="Some text")]
        monkeypatch.setattr(extractor_mod, "load_pdf_pages", lambda source: mock_pages)

        mock_client = MockOllamaClient(["NO_RECOMMENDATIONS_FOUND"])

        strategy = PromptStrategy(name="zero_shot", scheme=GRADE)
        result = extract_guideline("fake.pdf", strategy=strategy, client=mock_client)

        assert result.n_final_recommendations == 0
        assert result.recommendations_df.empty


class TestChunkSpans:
    @pytest.mark.parametrize("n_pages, pages_per_chunk", [(4, 1), (5, 3), (3, 2), (1, 3)])
    def test_spans_match_chunk_pages_text(self, n_pages, pages_per_chunk):
        from extraction.extractor import _chunk_pages, _chunk_spans
        from extraction.pdf_loader import PDFPage
        pages = [PDFPage(page_number=i + 1, text=f"text {i + 1}") for i in range(n_pages)]

        spans = _chunk_spans(pages, pages_per_chunk=pages_per_chunk)

        assert [t for _, t in spans] == _chunk_pages(pages, pages_per_chunk=pages_per_chunk)

    def test_span_start_pages_with_overlap(self):
        from extraction.extractor import _chunk_spans
        from extraction.pdf_loader import PDFPage
        pages = [PDFPage(page_number=i + 1, text=f"text {i + 1}") for i in range(5)]

        assert [p for p, _ in _chunk_spans(pages, pages_per_chunk=3, overlap=1)] == [1, 3]


def test_rows_carry_chunk_start_page(monkeypatch):
    """Synthetic 3-page doc, 2 pages per chunk: chunks start on pages 1 and 2."""
    import extraction.extractor as extractor_mod
    from tests.legacy_golden import MockClient, load_fixture
    from extraction.pdf_loader import PDFPage
    fx = load_fixture("synthetic")
    pages = [PDFPage(page_number=i + 1, text=t) for i, t in enumerate(fx["pages"])]
    monkeypatch.setattr(extractor_mod, "load_pdf_pages", lambda source: pages)

    result = extract_guideline("fake.pdf", strategy=PromptStrategy(name="zero_shot", scheme=GRADE),
                               client=MockClient(fx["responses"]), pages_per_chunk=fx["pages_per_chunk"])

    by_text = result.recommendations_df.set_index("recommendation")
    assert by_text.loc["We recommend early antibiotics", ["page", "chunk_id"]].tolist() == [1, 0]
    # Repeated in both chunks → earliest page wins
    assert by_text.loc["We suggest balanced crystalloids over saline", "page"] == 1
    assert by_text.loc["Reassess volume status frequently", ["page", "chunk_id"]].tolist() == [2, 1]


@pytest.mark.parametrize("seed, expected", [(None, "unset"), (5, 5)])
def test_extract_guideline_forwards_seed_only_when_set(monkeypatch, seed, expected):
    import extraction.extractor as extractor_mod
    from extraction.pdf_loader import PDFPage
    monkeypatch.setattr(extractor_mod, "load_pdf_pages", lambda source: [PDFPage(page_number=1, text="t")])
    seen = []

    class _Client(MockOllamaClient):
        def generate(self, prompt, num_ctx=None, **kwargs):
            seen.append(kwargs.get("seed", "unset"))
            return super().generate(prompt, num_ctx=num_ctx)

    extract_guideline("fake.pdf", strategy=PromptStrategy(name="zero_shot", scheme=GRADE),
                      client=_Client(["NO_RECOMMENDATIONS_FOUND"]), seed=seed)

    assert seen == [expected]
