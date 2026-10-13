"""The Paper 1 draft template types no number by hand (spec §7, §9 criterion 8).

Every number in the manuscript is a {{ field }} of the package exports; literal text may only hold
labels (Table 1, Figure 4), citation years, "95%" and "SHA-256".
"""

import os
import re

_TEMPLATE = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "paper1", "draft.md.j2")
_JINJA = re.compile(r"\{\{.*?\}\}|\{%.*?%\}|\{#.*?#\}", re.S)
_ALLOWED = (r"Table [12]\b", r"Table S1\b", r"Figure 4\b", r"JAMA 2019", r"95%", r"SHA-256")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def _literal_text() -> str:
    with open(_TEMPLATE, encoding="utf-8") as f:
        text = _JINJA.sub(" ", f.read())
    for pattern in _ALLOWED:
        text = re.sub(pattern, " ", text)
    return text


def test_no_hand_typed_numbers():
    numbers = [m.group(0) for m in _NUMBER.finditer(_literal_text())]
    assert numbers == [], f"numbers typed into the draft template: {numbers}"


def test_audit_catches_a_typed_number():
    text = _JINJA.sub(" ", "Strong was 26.1% ({{ t.ALL.strong_pct }}) in F1 and v3")
    assert [m.group(0) for m in _NUMBER.finditer(text)] == ["26.1"]
