"""One-off recorder for the legacy golden harness.

Run from the repo root against the code version to freeze:
    python tests/fixtures/legacy_golden/record.py

Writes <fixture>.json (page texts + canned LLM responses), expected_<run>.csv
and the V0 prompt golden strings. Page texts come from real PDFs, truncated.
"""

from __future__ import annotations

import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from evaluation.grading import GRADE  # noqa: E402
from extraction.pdf_loader import load_pdf_pages  # noqa: E402
from extraction.prompts import PromptStrategy, build_prompt  # noqa: E402
from tests.legacy_golden import GOLDEN_DIR, LEGACY_COLUMNS, RUNS, expected_path, run_case  # noqa: E402

_PDF_DIR = "/mnt/data1/klug/datasets/evidence_extraction/pdfs"
_N_PAGES = 3
_PAGE_CHARS = 1500

_EXAMPLES = [
    {"recommendation": "ACP recommends that clinicians use bisphosphonates for initial pharmacologic treatment",
     "class": "Strong For", "LOE": "High"},
    {"recommendation": "We suggest against the routine use of albumin for fluid resuscitation",
     "class": "Weak Against", "LOE": "Low"},
]

_ACP_R1 = ("ACP recommends monotherapy with either cognitive behavioral therapy or a second-generation "
           "antidepressant as initial treatment in patients in the acute phase of moderate to severe "
           "major depressive disorder")
_ACP_R2 = ("ACP suggests combination therapy with cognitive behavioral therapy and a second-generation "
           "antidepressant as initial treatment in patients in the acute phase of moderate to severe "
           "major depressive disorder")
_ACP_R3 = ("ACP suggests monotherapy with cognitive behavioral therapy as initial treatment in patients "
           "in the acute phase of mild major depressive disorder")

_ACP_RESPONSES = [
    # preamble + numbered rows + markdown bold
    f"Here are the extracted recommendations:\n1. {_ACP_R1} | strong recommendation | moderate-certainty evidence\n"
    f"2. **{_ACP_R2}** | conditional recommendation | low-certainty evidence",
    # case/space-variant duplicate, malformed line, truncated GT-like labels
    f"{_ACP_R1.upper()}  | Strong | Moderate\nthis line has no delimiters\n"
    f"{_ACP_R3} | onditional | Low certainty",
    "NO_RECOMMENDATIONS_FOUND",
]

_ACP_SC_RESPONSES = [
    _ACP_RESPONSES,
    [f"{_ACP_R1} | Strong | Moderate", f"{_ACP_R3} | Conditional | Low", "NO_RECOMMENDATIONS_FOUND"],
    [f"{_ACP_R1} (see text) | strong recommendation | high", "Background statement only | Weak | Low",
     "NO_RECOMMENDATIONS_FOUND"],
]

_ICU_RESPONSES = [
    "NO_RECOMMENDATIONS_FOUND",
    "| recommendation | grade | level |\n|---|---|---|\n"
    "| We suggest against the routine use of ultrafiltration in critically ill adults | "
    "Conditional recommendation, against | low certainty of evidence |\n"
    "| We cannot recommend for or against the use of diuretics to remove fluid | No recommendation | insufficient |",
    "We recommend fluid removal in patients with fluid overload | Strong recommendation for | moderate certainty",
]

_SYNTHETIC_PAGES = [
    "1. Introduction. Sepsis is common. Recommendation 1: We recommend early antibiotics (strong; high).",
    "Recommendation 2: We suggest balanced crystalloids over saline (conditional; moderate).",
    "Recommendation 3: Best practice statement: reassess volume status frequently.",
]

_SYNTHETIC_RESPONSES = [
    # chunk covers pages 1-2
    "We recommend early antibiotics | Strong For | High\n"
    "We suggest balanced crystalloids over saline | Weak For | Moderate",
    # chunk covers pages 2-3 (overlap repeats rec 2)
    "We suggest balanced crystalloids over saline | Weak For | Moderate\n"
    "Reassess volume status frequently | Best practice | NA",
]


def _pdf_pages(doi_filename: str) -> list[str]:
    pages = load_pdf_pages(os.path.join(_PDF_DIR, doi_filename + ".pdf"))
    return [p.text[:_PAGE_CHARS] for p in pages[:_N_PAGES]]


def _write_json(name: str, data: dict) -> None:
    with open(os.path.join(GOLDEN_DIR, f"{name}.json"), "w") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)


def _write_fixtures() -> None:
    _write_json("acp", {
        "pages": _pdf_pages("10_7326_M22-2056"), "pages_per_chunk": 1, "examples": _EXAMPLES,
        "responses": _ACP_RESPONSES, "sc_responses": _ACP_SC_RESPONSES,
    })
    _write_json("icu", {
        "pages": _pdf_pages("10_1007_s00134-025-08058-x"), "pages_per_chunk": 1, "examples": _EXAMPLES,
        "responses": _ICU_RESPONSES,
    })
    _write_json("synthetic", {
        "pages": _SYNTHETIC_PAGES, "pages_per_chunk": 2, "examples": _EXAMPLES,
        "responses": _SYNTHETIC_RESPONSES,
    })


def _write_prompt_goldens(acp_pages: list[str]) -> None:
    few_shot = PromptStrategy(name="few_shot", scheme=GRADE, examples=_EXAMPLES)
    zero_shot = PromptStrategy(name="zero_shot", scheme=GRADE)
    for name, strategy in (("prompt_v0_fewshot.txt", few_shot), ("prompt_v0_zeroshot.txt", zero_shot)):
        with open(os.path.join(GOLDEN_DIR, name), "w") as f:
            f.write(build_prompt(acp_pages[0], strategy))


class _Patch:
    """Minimal stand-in for pytest's monkeypatch outside pytest."""

    def setattr(self, obj, name, value):
        setattr(obj, name, value)


def main() -> None:
    _write_fixtures()
    with open(os.path.join(GOLDEN_DIR, "acp.json")) as f:
        _write_prompt_goldens(json.load(f)["pages"])

    for run_name, *_ in RUNS:
        df = run_case(run_name, _Patch())
        df[LEGACY_COLUMNS].to_csv(expected_path(run_name), index=False)
        print(f"{run_name}: {len(df)} rows")


if __name__ == "__main__":
    main()
