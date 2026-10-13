"""Scheme-adaptive prompt templates for recommendation extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from evaluation.grading import GradingScheme, GRADE, ABCD_123


# Maps grading schemes to domain-specific terminology for prompts
_SCHEME_TERMINOLOGY = {
    "grade": {
        "grade_label": "strength of recommendation",
        "level_label": "level of confidence",
        "grade_values": "strong/conditional [for/against]",
        "level_values": "high/moderate/low/very low",
        "domain": "clinical practice guideline",
    },
    "abcd_123": {
        "grade_label": "grade of recommendation",
        "level_label": "level of evidence",
        "grade_values": "A/B/C/D",
        "level_values": "1/2/3",
        "domain": "clinical practice guideline",
    },
    "esc_ers": {
        "grade_label": "class of recommendation",
        "level_label": "level of evidence",
        "grade_values": "I/IIa/IIb/III",
        "level_values": "A/B/C",
        "domain": "clinical practice guideline",
    },
}


class PromptVersion(str, Enum):
    V0 = "v0"  # legacy: graded recommendations only (text frozen, golden-tested)
    V1 = "v1"  # also best-practice statements and explicit non-recommendations


# Sentinel grade tokens for ungraded statements (V1); parsed into `category`
BEST_PRACTICE_TOKEN = "BEST_PRACTICE"
NO_RECOMMENDATION_TOKEN = "NO_RECOMMENDATION"
UNGRADED_LEVEL_TOKEN = "NA"


@dataclass
class PromptStrategy:
    """Configuration for a prompting strategy."""
    name: str  # "zero_shot" or "few_shot"
    scheme: GradingScheme
    examples: list[dict] = field(default_factory=list)  # For few-shot: [{recommendation, class, LOE}]
    prompt_version: PromptVersion = PromptVersion.V0


def build_prompt(
    page_text: str,
    strategy: PromptStrategy,
    output_format: str = "pipe",
) -> str:
    """Build a complete prompt for extracting recommendations from a guideline text.

    Args:
        page_text: The guideline text to extract from.
        strategy: Prompting strategy (zero_shot or few_shot with scheme).
        output_format: "pipe" for pipe-delimited or "json" for JSON array.
    """
    scheme_name = strategy.scheme.name
    terms = _SCHEME_TERMINOLOGY.get(scheme_name)
    if terms is None:
        raise ValueError(f"No terminology defined for scheme '{scheme_name}'")

    grade_label = terms["grade_label"]
    level_label = terms["level_label"]
    grade_values = terms["grade_values"]
    level_values = terms["level_values"]

    parts = []
    is_v1 = strategy.prompt_version == PromptVersion.V1

    # V1 widens the definition to ungraded statements; V0 text must stay byte-identical
    if is_v1:
        definition = (
            f"A \"recommendation\" is an actionable statement that directs clinical practice. "
            f"It is usually graded with a {grade_label} and a {level_label}; ungraded "
            f"best-practice statements and explicit non-recommendations also count.\n"
        )
    else:
        definition = (
            f"A \"recommendation\" is an actionable statement that directs clinical practice "
            f"and is explicitly graded with a {grade_label} and a {level_label}.\n"
        )

    # System instruction with precise definition
    parts.append(
        f"You are an expert medical researcher. Extract all clinical recommendations "
        f"from the following {terms['domain']} text.\n"
        f"\n"
        f"{definition}"
        f"\n"
        f"For each recommendation, extract:\n"
        f"- The recommendation text — copy it EXACTLY as written in the source\n"
        f"- The {grade_label} — use ONLY these valid values: {grade_values}\n"
        f"- The {level_label} — use ONLY these valid values: {level_values}\n"
    )

    # Output format
    if output_format == "json":
        parts.append(
            f"Output a JSON array of objects, each with keys \"text\", \"grade\", \"level\".\n"
            f"Example: {{\"recommendations\": [{{\"text\": \"...\", \"grade\": \"...\", \"level\": \"...\"}}]}}\n"
        )
    else:
        parts.append(
            f"Output format: one recommendation per line, using pipe delimiters:\n"
            f"recommendation text | {grade_label} | {level_label}\n"
        )

    # Rules with negative examples
    if is_v1:
        scope_rules = (
            f"- Extract ONLY explicitly stated recommendations\n"
            f"- Copy the recommendation text EXACTLY as written — do NOT paraphrase or summarize\n"
            f"- Also extract best-practice / good-practice statements: write {BEST_PRACTICE_TOKEN} "
            f"as the {grade_label} and {UNGRADED_LEVEL_TOKEN} as the {level_label}\n"
            f"- Also extract explicit non-recommendations (\"we cannot recommend for or against\", "
            f"\"evidence was insufficient to make a recommendation\"): write {NO_RECOMMENDATION_TOKEN} "
            f"as the {grade_label} and {UNGRADED_LEVEL_TOKEN} as the {level_label}\n"
            f"- Do NOT extract: background statements, evidence summaries, section headers\n"
        )
    else:
        scope_rules = (
            f"- Extract ONLY explicitly stated recommendations with a clear {grade_label} and {level_label}\n"
            f"- Copy the recommendation text EXACTLY as written — do NOT paraphrase or summarize\n"
            f"- Do NOT extract: background statements, evidence summaries, section headers, "
            f"or statements without an explicit {grade_label} and {level_label}\n"
        )
    parts.append(
        f"Rules:\n"
        f"{scope_rules}"
        f"- Do NOT infer or create recommendations that are not in the text\n"
        f"- If no recommendations are found, output exactly: NO_RECOMMENDATIONS_FOUND\n"
        f"- Do NOT include headers, row numbers, or any other text"
    )

    # Few-shot examples
    if strategy.name == "few_shot" and strategy.examples:
        parts.append("\nExamples of correctly extracted recommendations:")
        for ex in strategy.examples:
            source = ex.get("source_text")
            if source:
                parts.append(f"Source: \"{source}\"")
            parts.append(f"{ex['recommendation']} | {ex['class']} | {ex['LOE']}")
        parts.append("")

    # The actual text
    parts.append(f"\n--- Guideline Text ---\n{page_text}\n--- End of Text ---")
    parts.append(f"\nExtracted recommendations:")

    return "\n".join(parts)
