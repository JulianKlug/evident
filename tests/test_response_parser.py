"""Tests for extraction.response_parser."""

import pandas as pd
import pytest

from extraction.response_parser import (
    parse_llm_response,
    _strip_thinking_tags,
    _remove_boilerplate,
    _parse_pipe_delimited,
    _fallback_csv_parse,
    NO_RECOMMENDATIONS_SENTINEL,
)


class TestStripThinkingTags:
    def test_removes_think_block(self):
        text = "<think>Let me analyze this page...</think>rec1 | strong | high"
        assert "<think>" not in _strip_thinking_tags(text)
        assert "rec1" in _strip_thinking_tags(text)

    def test_removes_multiline_think(self):
        text = "<think>\nThis is a complex page.\nI need to think.\n</think>\nrec1 | A | 1"
        result = _strip_thinking_tags(text)
        assert "<think>" not in result
        assert "rec1 | A | 1" in result

    def test_no_think_tags(self):
        text = "rec1 | strong | high"
        assert _strip_thinking_tags(text) == text

    def test_multiple_think_blocks(self):
        text = "<think>first</think>rec1 | A | 1\n<think>second</think>rec2 | B | 2"
        result = _strip_thinking_tags(text)
        assert "rec1" in result
        assert "rec2" in result
        assert "<think>" not in result


class TestRemoveBoilerplate:
    def test_strips_preamble(self):
        text = "Here are the recommendations I found:\n\nrec1 | strong | high\nrec2 | weak | low"
        result = _remove_boilerplate(text)
        assert "rec1 | strong | high" in result
        assert "rec2 | weak | low" in result

    def test_keeps_pipe_lines(self):
        text = "rec1 | A | 1\nrec2 | B | 2"
        result = _remove_boilerplate(text)
        assert "rec1 | A | 1" in result


class TestParsePipeDelimited:
    def test_basic_pipe_parsing(self):
        text = "Use beta-blockers for HF | strong | high\nUse ACE for HTN | weak | moderate"
        records = _parse_pipe_delimited(text)
        assert len(records) == 2
        assert records[0]["recommendation"] == "Use beta-blockers for HF"
        assert records[0]["class"] == "strong"
        assert records[0]["LOE"] == "high"

    def test_skips_header_line(self):
        text = "Recommendation | Class | LOE\nrec1 | A | 1"
        records = _parse_pipe_delimited(text)
        assert len(records) == 1
        assert records[0]["recommendation"] == "rec1"

    def test_skips_separator_line(self):
        text = "rec1 | A | 1\n---|---|---\nrec2 | B | 2"
        records = _parse_pipe_delimited(text)
        assert len(records) == 2

    def test_strips_row_numbers(self):
        text = "1. rec1 | A | 1\n2. rec2 | B | 2"
        records = _parse_pipe_delimited(text)
        assert len(records) == 2
        assert records[0]["recommendation"] == "rec1"

    def test_handles_markdown_table_format(self):
        text = "| rec1 | A | 1 |\n| rec2 | B | 2 |"
        records = _parse_pipe_delimited(text)
        assert len(records) == 2
        assert records[0]["recommendation"] == "rec1"

    def test_empty_input(self):
        assert _parse_pipe_delimited("") == []
        assert _parse_pipe_delimited("   \n  \n") == []

    def test_wrong_field_count(self):
        text = "only two fields | here\nrec1 | A | 1"
        records = _parse_pipe_delimited(text)
        assert len(records) == 1
        assert records[0]["recommendation"] == "rec1"


class TestFallbackCSVParse:
    def test_basic_csv(self):
        text = '"Use beta-blockers for HF",strong,high\n"Use ACE for HTN",weak,moderate'
        records = _fallback_csv_parse(text)
        assert len(records) == 2
        assert records[0]["class"] == "strong"

    def test_csv_with_row_numbers(self):
        text = '1,"rec1",A,1\n2,"rec2",B,2'
        records = _fallback_csv_parse(text)
        assert len(records) == 2
        assert records[0]["recommendation"] == "rec1"

    def test_csv_with_commas_in_text(self):
        text = '"First, use beta-blockers",strong,high'
        records = _fallback_csv_parse(text)
        assert len(records) == 1
        assert "beta-blockers" in records[0]["recommendation"]


class TestParseLLMResponse:
    def test_full_pipeline_pipe(self):
        text = "rec1 | strong for | high\nrec2 | weak for | low"
        df = parse_llm_response(text)
        assert len(df) == 2
        assert list(df.columns) == ["recommendation", "class", "LOE"]

    def test_with_thinking(self):
        text = "<think>Analyzing the page...</think>\nrec1 | A | 1"
        df = parse_llm_response(text, has_thinking=True)
        assert len(df) == 1

    def test_no_recommendations_sentinel(self):
        text = "NO_RECOMMENDATIONS_FOUND"
        df = parse_llm_response(text)
        assert df.empty

    def test_empty_input(self):
        assert parse_llm_response("").empty
        assert parse_llm_response("   ").empty
        assert parse_llm_response(None).empty

    def test_preamble_stripped(self):
        text = "I found the following recommendations:\n\nrec1 | A | 1\nrec2 | B | 2\n\nThese are all the recommendations."
        df = parse_llm_response(text)
        assert len(df) == 2
