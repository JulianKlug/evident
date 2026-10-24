"""The Paper 1 draft template types no number by hand (spec §7, §9 criterion 8).

The draft is private (not tracked); the audit runs where it exists and on the test stand-in.

Every number in the manuscript is a {{ field }} of the package exports; literal text may only hold
labels (Table 1, Figure 4, eTable 2), "JAMA 2019", "95%" and "SHA-256".
"""

import os
import re

import pytest

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
_TEMPLATES = [os.path.join(_ROOT, "docs", "paper1", "draft.md.j2"),  # private, absent in a clean checkout
              os.path.join(_ROOT, "tests", "fixtures", "paper", "draft.md.j2")]
_JINJA = re.compile(r"\{\{.*?\}\}|\{%.*?%\}|\{#.*?#\}", re.S)
_ALLOWED = (r"Table [12]\b", r"Table S1\b", r"Figure [1-4]\b", r"e(?:Table|Figure|Appendix) \d\b", r"JAMA 2019",
            r"95%", r"SHA-256")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def _literal_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        text = _JINJA.sub(" ", f.read())
    for pattern in _ALLOWED:
        text = re.sub(pattern, " ", text)
    return text


@pytest.mark.parametrize("path", _TEMPLATES)
def test_no_hand_typed_numbers(path):
    if not os.path.isfile(path):
        pytest.skip(f"{path} not present")
    numbers = [m.group(0) for m in _NUMBER.finditer(_literal_text(path))]
    assert numbers == [], f"numbers typed into the draft template: {numbers}"


def test_audit_catches_a_typed_number():
    text = _JINJA.sub(" ", "Strong was 26.1% ({{ t.ALL.strong_pct }}) in F1 and v3")
    assert [m.group(0) for m in _NUMBER.finditer(text)] == ["26.1"]
