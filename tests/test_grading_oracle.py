"""Tests for extraction.grading_oracle."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from extraction.grading_oracle import (
    _strip_thinking,
    _parse_regrade_response,
    regrade_recommendations,
)
from extraction.llm_client import LLMResponse
from evaluation.grading import GRADE, ABCD_123, GradingScheme


# ── Unit tests for _strip_thinking ──────────────────────────────────────────


class TestStripThinking:
    def test_removes_think_block(self):
        text = "<think>some reasoning here</think>\ngrade: Strong For"
        assert _strip_thinking(text) == "grade: Strong For"

    def test_removes_multiline_think_block(self):
        text = "<think>\nlong\nreasoning\n</think>\nstrength of recommendation: Strong For\nlevel of confidence: High"
        result = _strip_thinking(text)
        assert "<think>" not in result
        assert "strength of recommendation: Strong For" in result

    def test_no_think_block(self):
        text = "strength of recommendation: Strong For\nlevel of confidence: High"
        assert _strip_thinking(text) == text

    def test_empty_string(self):
        assert _strip_thinking("") == ""


# ── Unit tests for _parse_regrade_response ──────────────────────────────────


class TestParseRegradeResponse:
    def test_grade_scheme_valid(self):
        text = "strength of recommendation: Strong For\nlevel of confidence: High"
        grade, level = _parse_regrade_response(
            text,
            grade_label="strength of recommendation",
            level_label="level of confidence",
            valid_grades={"Strong For", "Conditional For", "Strong Against", "Conditional Against"},
            valid_levels={"High", "Moderate", "Low", "Very Low"},
        )
        assert grade == "Strong For"
        assert level == "High"

    def test_case_insensitive_matching(self):
        text = "strength of recommendation: strong for\nlevel of confidence: high"
        grade, level = _parse_regrade_response(
            text,
            grade_label="strength of recommendation",
            level_label="level of confidence",
            valid_grades={"Strong For", "Conditional For"},
            valid_levels={"High", "Moderate", "Low"},
        )
        assert grade == "Strong For"  # canonical casing
        assert level == "High"

    def test_invalid_grade_returns_none(self):
        text = "strength of recommendation: InvalidGrade\nlevel of confidence: High"
        grade, level = _parse_regrade_response(
            text,
            grade_label="strength of recommendation",
            level_label="level of confidence",
            valid_grades={"Strong For", "Conditional For"},
            valid_levels={"High", "Moderate"},
        )
        assert grade is None
        assert level == "High"

    def test_invalid_level_returns_none(self):
        text = "strength of recommendation: Strong For\nlevel of confidence: Unknown"
        grade, level = _parse_regrade_response(
            text,
            grade_label="strength of recommendation",
            level_label="level of confidence",
            valid_grades={"Strong For"},
            valid_levels={"High", "Moderate"},
        )
        assert grade == "Strong For"
        assert level is None

    def test_unparseable_response(self):
        text = "I don't know what to do with this recommendation."
        grade, level = _parse_regrade_response(
            text,
            grade_label="strength of recommendation",
            level_label="level of confidence",
            valid_grades={"Strong For"},
            valid_levels={"High"},
        )
        assert grade is None
        assert level is None

    def test_with_thinking_tags(self):
        text = "<think>Let me analyze...</think>\nstrength of recommendation: Conditional For\nlevel of confidence: Moderate"
        grade, level = _parse_regrade_response(
            text,
            grade_label="strength of recommendation",
            level_label="level of confidence",
            valid_grades={"Strong For", "Conditional For"},
            valid_levels={"High", "Moderate", "Low"},
        )
        assert grade == "Conditional For"
        assert level == "Moderate"

    def test_abcd_scheme(self):
        text = "grade of recommendation: B\nlevel of evidence: 2"
        grade, level = _parse_regrade_response(
            text,
            grade_label="grade of recommendation",
            level_label="level of evidence",
            valid_grades={"A", "B", "C", "D"},
            valid_levels={"1", "2", "3"},
        )
        assert grade == "B"
        assert level == "2"

    def test_esc_ers_scheme(self):
        text = "class of recommendation: IIa\nlevel of evidence: B"
        grade, level = _parse_regrade_response(
            text,
            grade_label="class of recommendation",
            level_label="level of evidence",
            valid_grades={"I", "IIa", "IIb", "III"},
            valid_levels={"A", "B", "C"},
        )
        assert grade == "IIa"
        assert level == "B"

    def test_extra_whitespace(self):
        text = "  strength of recommendation:  Strong For  \n  level of confidence:  High  "
        grade, level = _parse_regrade_response(
            text,
            grade_label="strength of recommendation",
            level_label="level of confidence",
            valid_grades={"Strong For"},
            valid_levels={"High"},
        )
        assert grade == "Strong For"
        assert level == "High"


# ── Integration tests for regrade_recommendations ──────────────────────────


class TestRegradeRecommendations:
    def _make_mock_client_class(self, responses):
        """Create a mock OllamaClient class that returns canned responses."""
        call_count = [0]

        class MockClient:
            def __init__(self, model="mock"):
                self.model = model

            def generate(self, prompt, **kwargs):
                idx = call_count[0]
                call_count[0] += 1
                text = responses[idx] if idx < len(responses) else ""
                return LLMResponse(
                    raw_text=text, model="mock",
                    prompt_tokens=100, eval_tokens=50,
                    total_duration_ms=10.0,
                )

        return MockClient

    def test_updates_incorrect_grade(self):
        """Oracle corrects an incorrect grade."""
        df = pd.DataFrame({
            "recommendation": ["Use beta-blockers for heart failure"],
            "class": ["Conditional For"],
            "LOE": ["High"],
        })
        responses = ["strength of recommendation: Strong For\nlevel of confidence: High"]
        MockClient = self._make_mock_client_class(responses)

        with patch("extraction.grading_oracle.OllamaClient", MockClient):
            result = regrade_recommendations(df, GRADE)

        assert result.iloc[0]["class"] == "Strong For"
        assert result.iloc[0]["LOE"] == "High"

    def test_updates_incorrect_level(self):
        """Oracle corrects an incorrect level."""
        df = pd.DataFrame({
            "recommendation": ["Use ACE inhibitors for HTN"],
            "class": ["Strong For"],
            "LOE": ["High"],
        })
        responses = ["strength of recommendation: Strong For\nlevel of confidence: Moderate"]
        MockClient = self._make_mock_client_class(responses)

        with patch("extraction.grading_oracle.OllamaClient", MockClient):
            result = regrade_recommendations(df, GRADE)

        assert result.iloc[0]["class"] == "Strong For"  # unchanged
        assert result.iloc[0]["LOE"] == "Moderate"  # updated

    def test_keeps_original_on_parse_failure(self):
        """Fail-safe: unparseable response leaves values unchanged."""
        df = pd.DataFrame({
            "recommendation": ["Use statins for cholesterol"],
            "class": ["Strong For"],
            "LOE": ["High"],
        })
        responses = ["I cannot determine the correct grading for this."]
        MockClient = self._make_mock_client_class(responses)

        with patch("extraction.grading_oracle.OllamaClient", MockClient):
            result = regrade_recommendations(df, GRADE)

        assert result.iloc[0]["class"] == "Strong For"
        assert result.iloc[0]["LOE"] == "High"

    def test_keeps_original_on_invalid_value(self):
        """Fail-safe: invalid value in response leaves field unchanged."""
        df = pd.DataFrame({
            "recommendation": ["Use diuretics"],
            "class": ["Strong For"],
            "LOE": ["High"],
        })
        responses = ["strength of recommendation: SuperStrong\nlevel of confidence: High"]
        MockClient = self._make_mock_client_class(responses)

        with patch("extraction.grading_oracle.OllamaClient", MockClient):
            result = regrade_recommendations(df, GRADE)

        assert result.iloc[0]["class"] == "Strong For"  # unchanged (invalid value rejected)
        assert result.iloc[0]["LOE"] == "High"

    def test_empty_dataframe(self):
        """Empty DataFrame passes through unchanged."""
        df = pd.DataFrame(columns=["recommendation", "class", "LOE"])
        result = regrade_recommendations(df, GRADE)
        assert result.empty

    def test_multiple_rows(self):
        """Multiple rows are each processed independently."""
        df = pd.DataFrame({
            "recommendation": [
                "Use beta-blockers for HF",
                "Use ACE inhibitors for HTN",
                "Use statins for cholesterol",
            ],
            "class": ["Conditional For", "Strong For", "Conditional For"],
            "LOE": ["High", "High", "Low"],
        })
        responses = [
            "strength of recommendation: Strong For\nlevel of confidence: High",
            "strength of recommendation: Strong For\nlevel of confidence: Moderate",
            "garbage response that wont parse",
        ]
        MockClient = self._make_mock_client_class(responses)

        with patch("extraction.grading_oracle.OllamaClient", MockClient):
            result = regrade_recommendations(df, GRADE)

        assert len(result) == 3
        # Row 0: grade corrected
        assert result.iloc[0]["class"] == "Strong For"
        assert result.iloc[0]["LOE"] == "High"
        # Row 1: level corrected
        assert result.iloc[1]["class"] == "Strong For"
        assert result.iloc[1]["LOE"] == "Moderate"
        # Row 2: unchanged (parse failure)
        assert result.iloc[2]["class"] == "Conditional For"
        assert result.iloc[2]["LOE"] == "Low"

    def test_does_not_mutate_input(self):
        """regrade_recommendations should not mutate the input DataFrame."""
        df = pd.DataFrame({
            "recommendation": ["Use beta-blockers"],
            "class": ["Conditional For"],
            "LOE": ["High"],
        })
        original_grade = df.iloc[0]["class"]
        responses = ["strength of recommendation: Strong For\nlevel of confidence: High"]
        MockClient = self._make_mock_client_class(responses)

        with patch("extraction.grading_oracle.OllamaClient", MockClient):
            result = regrade_recommendations(df, GRADE)

        assert df.iloc[0]["class"] == original_grade  # input unchanged

    def test_with_thinking_tags_in_response(self):
        """deepseek-r1 thinking tags are stripped before parsing."""
        df = pd.DataFrame({
            "recommendation": ["Use beta-blockers for HF"],
            "class": ["Conditional For"],
            "LOE": ["Low"],
        })
        responses = [
            "<think>\nThe recommendation clearly states beta-blockers for HF.\n"
            "The evidence is strong.\n</think>\n"
            "strength of recommendation: Strong For\n"
            "level of confidence: High"
        ]
        MockClient = self._make_mock_client_class(responses)

        with patch("extraction.grading_oracle.OllamaClient", MockClient):
            result = regrade_recommendations(df, GRADE)

        assert result.iloc[0]["class"] == "Strong For"
        assert result.iloc[0]["LOE"] == "High"

    def test_abcd_scheme(self):
        """Works with ABCD_123 grading scheme."""
        df = pd.DataFrame({
            "recommendation": ["Give aspirin"],
            "class": ["C"],
            "LOE": ["3"],
        })
        responses = ["grade of recommendation: A\nlevel of evidence: 1"]
        MockClient = self._make_mock_client_class(responses)

        with patch("extraction.grading_oracle.OllamaClient", MockClient):
            result = regrade_recommendations(df, ABCD_123)

        assert result.iloc[0]["class"] == "A"
        assert result.iloc[0]["LOE"] == "1"

    def test_custom_model(self):
        """The model parameter is passed to OllamaClient."""
        df = pd.DataFrame({
            "recommendation": ["Use beta-blockers"],
            "class": ["Strong For"],
            "LOE": ["High"],
        })
        responses = ["strength of recommendation: Strong For\nlevel of confidence: High"]
        captured_models = []

        class TrackingClient:
            def __init__(self, model="mock"):
                self.model = model
                captured_models.append(model)

            def generate(self, prompt, **kwargs):
                return LLMResponse(
                    raw_text=responses[0], model=self.model,
                    prompt_tokens=100, eval_tokens=50,
                    total_duration_ms=10.0,
                )

        with patch("extraction.grading_oracle.OllamaClient", TrackingClient):
            regrade_recommendations(df, GRADE, model="custom-model:7b")

        assert captured_models == ["custom-model:7b"]
