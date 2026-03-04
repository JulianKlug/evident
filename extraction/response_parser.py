"""Parse LLM output into structured recommendation DataFrames."""

from __future__ import annotations

import csv
import io
import re

import pandas as pd


_THINKING_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
_ROW_NUMBER_PATTERN = re.compile(r"^\s*\d+[\.\)]\s*")
_HEADER_PATTERN = re.compile(
    r"^\s*(recommendation|#|no\.?|number)\s*\|",
    re.IGNORECASE,
)
_PIPE_LINE_PATTERN = re.compile(r"^(.+?)\|(.+?)\|(.+)$")

NO_RECOMMENDATIONS_SENTINEL = "NO_RECOMMENDATIONS_FOUND"


def parse_llm_response(raw_text: str, has_thinking: bool = False) -> pd.DataFrame:
    """Parse raw LLM text into a DataFrame with columns [recommendation, class, LOE].

    Pipeline: strip thinking → remove boilerplate → pipe-delimited parse → CSV fallback.
    """
    if not raw_text or not raw_text.strip():
        return pd.DataFrame(columns=["recommendation", "class", "LOE"])

    text = raw_text
    if has_thinking:
        text = _strip_thinking_tags(text)

    text = _remove_boilerplate(text)

    if NO_RECOMMENDATIONS_SENTINEL in text:
        return pd.DataFrame(columns=["recommendation", "class", "LOE"])

    # Try pipe-delimited first
    records = _parse_pipe_delimited(text)
    if not records:
        records = _fallback_csv_parse(text)

    if not records:
        return pd.DataFrame(columns=["recommendation", "class", "LOE"])

    return pd.DataFrame(records, columns=["recommendation", "class", "LOE"])


def _strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks produced by deepseek-r1."""
    return _THINKING_PATTERN.sub("", text).strip()


def _remove_boilerplate(text: str) -> str:
    """Strip preamble/postamble LLM commentary, keeping only the table data."""
    lines = text.strip().split("\n")
    cleaned = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        # Detect table start: line with pipes or starts with a number
        if "|" in stripped or _ROW_NUMBER_PATTERN.match(stripped):
            in_table = True
        if in_table:
            cleaned.append(line)
        # Also keep lines that look like CSV data (quoted or comma-separated with 3+ fields)
        elif not in_table and stripped.count(",") >= 2:
            cleaned.append(line)
            in_table = True

    if not cleaned:
        # If no table detected, return original (might be a single-line response)
        return text.strip()
    return "\n".join(cleaned)


def _parse_pipe_delimited(text: str) -> list[dict]:
    """Parse lines in format: recommendation | class | LOE."""
    records = []
    for line in text.strip().split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # Skip header lines
        if _HEADER_PATTERN.match(stripped):
            continue
        # Skip separator lines (e.g., ---|---|---)
        if re.match(r"^[\s\-\|:]+$", stripped):
            continue

        # Strip leading row numbers
        stripped = _ROW_NUMBER_PATTERN.sub("", stripped)

        # Strip leading/trailing pipes (markdown table format)
        if stripped.startswith("|"):
            stripped = stripped[1:]
        if stripped.endswith("|"):
            stripped = stripped[:-1]

        match = _PIPE_LINE_PATTERN.match(stripped)
        if match:
            rec = match.group(1).strip()
            grade = match.group(2).strip()
            loe = match.group(3).strip()
            if rec and grade and loe:
                records.append({
                    "recommendation": rec,
                    "class": grade,
                    "LOE": loe,
                })
    return records


def _fallback_csv_parse(text: str) -> list[dict]:
    """CSV fallback parser for when the model ignores pipe formatting instructions."""
    records = []
    try:
        reader = csv.reader(io.StringIO(text.strip()))
        for row in reader:
            if len(row) < 3:
                continue
            # Skip header rows
            if row[0].strip().lower() in ("recommendation", "#", "no", "number", "no."):
                continue
            # Strip leading row number if first field is numeric
            start_idx = 0
            if row[0].strip().isdigit():
                start_idx = 1
            remaining = row[start_idx:]
            if len(remaining) < 3:
                continue
            # Last two fields are class and LOE; everything before is the recommendation
            loe = remaining[-1].strip()
            grade = remaining[-2].strip()
            rec = ", ".join(f.strip() for f in remaining[:-2])
            if rec and grade and loe:
                records.append({
                    "recommendation": rec,
                    "class": grade,
                    "LOE": loe,
                })
    except csv.Error:
        pass
    return records
