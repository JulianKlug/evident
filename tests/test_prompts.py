"""Tests for extraction.prompts."""

import pytest

from evaluation.grading import GRADE, ABCD_123, ESC_ERS
from extraction.prompts import PromptStrategy, build_prompt


class TestBuildPrompt:
    def test_grade_scheme_zero_shot(self):
        strategy = PromptStrategy(name="zero_shot", scheme=GRADE)
        prompt = build_prompt("Some guideline text here.", strategy)
        assert "strength of recommendation" in prompt
        assert "level of confidence" in prompt
        assert "strong/conditional" in prompt
        assert "high/moderate/low/very low" in prompt
        assert "pipe delimiters" in prompt
        assert "NO_RECOMMENDATIONS_FOUND" in prompt
        assert "Some guideline text here." in prompt

    def test_abcd_scheme_zero_shot(self):
        strategy = PromptStrategy(name="zero_shot", scheme=ABCD_123)
        prompt = build_prompt("Page text.", strategy)
        assert "grade of recommendation" in prompt
        assert "level of evidence" in prompt
        assert "A/B/C/D" in prompt
        assert "1/2/3" in prompt

    def test_esc_ers_scheme(self):
        strategy = PromptStrategy(name="zero_shot", scheme=ESC_ERS)
        prompt = build_prompt("Page text.", strategy)
        assert "class of recommendation" in prompt
        assert "I/IIa/IIb/III" in prompt

    def test_few_shot_includes_examples(self):
        examples = [
            {"recommendation": "Use drug X for condition Y", "class": "Strong For", "LOE": "High"},
            {"recommendation": "Consider drug Z", "class": "Weak For", "LOE": "Low"},
        ]
        strategy = PromptStrategy(name="few_shot", scheme=GRADE, examples=examples)
        prompt = build_prompt("Page text.", strategy)
        assert "Use drug X for condition Y | Strong For | High" in prompt
        assert "Consider drug Z | Weak For | Low" in prompt
        assert "Examples" in prompt

    def test_zero_shot_no_examples(self):
        strategy = PromptStrategy(name="zero_shot", scheme=GRADE)
        prompt = build_prompt("Page text.", strategy)
        assert "Examples" not in prompt

    def test_unknown_scheme_raises(self):
        from evaluation.grading import GradingScheme
        custom = GradingScheme(name="custom_unknown", grades=["X"], levels=["Y"])
        strategy = PromptStrategy(name="zero_shot", scheme=custom)
        with pytest.raises(ValueError, match="No terminology"):
            build_prompt("text", strategy)
