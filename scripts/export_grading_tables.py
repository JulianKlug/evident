"""Export the reviewed grading crosswalk (xlsx) to the CSV tables evident reads, and render the mapping doc.

    out/grading_harmonisation/grading_systems_harmonisation_jk.xlsx   (reviewed by hand, provenance)
        │ sheets Scales + Crosswalk; grouped codes exploded ("2a / 2b / 2c" → 3 rows)
        ▼
    evident/grading_data/{scales,crosswalk,families}.csv              (source of truth for the code)
        │ --doc
        ▼
    docs/harmonization_tables.md                                      (generated; a test diffs it)

Usage:
    python scripts/export_grading_tables.py [--xlsx PATH]     # workbook → CSV
    python scripts/export_grading_tables.py --doc             # CSV → docs/harmonization_tables.md
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evident.grading_tables import (  # noqa: E402
    CROSSWALK_COLUMNS,
    DATA_DIR,
    DECISION_COLUMNS,
    FAMILY_COLUMNS,
    SCALE_COLUMNS,
    WILDCARD,
    render_doc,
)

_XLSX = "out/grading_harmonisation/grading_systems_harmonisation_jk.xlsx"
_DOC = "docs/harmonization_tables.md"
_FIRST_DATA_ROW = 4  # rows 1–3: title, blank, header
_GROUP_SEPARATOR = " / "

# Codes the generic " / " split cannot explode correctly, e.g. EFCNI "B, any level" → one row per printed level
_EXPLICIT_CODES = {
    ("E_NICE_OLD", "II (IIa/IIb)"): ["II", "IIa", "IIb"],
    ("E_NCCN", "Category 2A / 2B / 3"): ["Category 2A", "Category 2B", "Category 3"],
    ("E_EFCNI", "B, any level"): [f"B, {q} quality" for q in ("high", "moderate", "low", "very low")],
    ("E_EFCNI", "C, any level"): [f"C, {q} quality" for q in ("high", "moderate")],
    ("S_CONSENSUS", "(any agreement level)"): [WILDCARD],
    ("S_NONE", "(none)"): [WILDCARD],
    ("S_NO_GRADING", "(none)"): [WILDCARD],
    ("E_NONE", "(none)"): [WILDCARD],
}
# Printed variants of a code, e.g. EGS writes strength "I"/"II" for numeric GRADE "1"/"2"
_EXTRA_ALIASES = {
    ("S_GRADE_NUM", "1"): ["I"],
    ("S_GRADE_NUM", "2"): ["II"],
}
_PLACEHOLDER_SCALES = {"S_UNKNOWN", "E_UNKNOWN"}  # "to be determined": no mapping to export

# Family → default scale pair; a guideline overrides it in corpus/manifest.csv (decision D10)
_FAMILIES = [
    ("esc_class", "S_ESC_CLASS", "E_ESC"),
    ("acc_aha", "S_ACCAHA_COR", "E_ACCAHA"),
    ("idsa_esmo", "S_IDSA_AE", "E_ROMAN5_ESMO"),
    ("oxford", "S_LETTER_GRADE", "E_OXFORD2011"),
    ("sign", "S_LETTER_GRADE", "E_SIGN"),
    ("uspstf", "S_USPSTF", "E_GRADE4"),
    ("aan", "S_AAN", "E_AAN_CLASS"),
    ("awmf", "S_AWMF", "E_OXFORD2011"),
    ("grade_numeric", "S_GRADE_NUM", "E_GRADE_LETTER"),
    ("easl", "S_EASL3", "E_OXFORD2011"),
    ("nccn", "S_NCCN", "E_NCCN"),
    ("consensus_only", "S_NO_GRADING", "E_NONE"),
    ("other", "", ""),
]


def _codes(scale: str, code: str) -> list[str]:
    """'2a / 2b / 2c' → ['2a', '2b', '2c']; explicit overrides first."""
    if (scale, code) in _EXPLICIT_CODES:
        return _EXPLICIT_CODES[(scale, code)]
    return [c.strip() for c in code.split(_GROUP_SEPARATOR)]


def _text(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip() if value is not None else ""


def _read_workbook(path: str):
    import openpyxl  # pipeline extra; only this exporter needs it

    wb = openpyxl.load_workbook(path, read_only=True)
    scales = [r for r in wb["Scales"].iter_rows(min_row=_FIRST_DATA_ROW, values_only=True) if r[0]]
    crosswalk = [r for r in wb["Crosswalk"].iter_rows(min_row=_FIRST_DATA_ROW, values_only=True) if r[0]]
    decisions = [r for r in wb["Decisions to validate"].iter_rows(min_row=_FIRST_DATA_ROW, values_only=True)
                 if r[0]]
    return scales, crosswalk, decisions


def _write(path: str, columns, rows) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def export(xlsx: str) -> None:
    scales, crosswalk, decisions = _read_workbook(xlsx)

    scale_rows = [(_text(key), _text(axis).lower(), _text(desc), _text(quality))
                  for key, axis, desc, quality, *_ in scales if key not in _PLACEHOLDER_SCALES]

    # One row per native code; equivalent codes of a group share the mapping
    crosswalk_rows = []
    for scale, axis, code, meaning, value, direction, alternative, confidence, note in crosswalk:
        if scale in _PLACEHOLDER_SCALES:
            continue
        for native in _codes(scale, _text(code)):
            aliases = "|".join(_EXTRA_ALIASES.get((scale, native), []))
            crosswalk_rows.append((scale, _text(axis).lower(), native, aliases, _text(meaning), _text(value),
                                   _text(direction), _text(alternative), _text(confidence), _text(note)))

    os.makedirs(DATA_DIR, exist_ok=True)
    _write(os.path.join(DATA_DIR, "scales.csv"), SCALE_COLUMNS, scale_rows)
    _write(os.path.join(DATA_DIR, "crosswalk.csv"), CROSSWALK_COLUMNS, crosswalk_rows)
    _write(os.path.join(DATA_DIR, "families.csv"), FAMILY_COLUMNS, _FAMILIES)
    _write(os.path.join(DATA_DIR, "decisions.csv"), DECISION_COLUMNS,
           [(_text(i), _text(q), _text(rule), _text(alt), _text(why)) for i, q, rule, alt, why, *_ in decisions])
    print(f"{len(scale_rows)} scales, {len(crosswalk_rows)} crosswalk rows → {DATA_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xlsx", default=_XLSX)
    parser.add_argument("--doc", action="store_true", help="render docs/harmonization_tables.md from the CSVs")
    args = parser.parse_args()

    if args.doc:
        with open(_DOC, "w") as f:
            f.write(render_doc())
        print(f"wrote {_DOC}")
        return

    export(args.xlsx)


if __name__ == "__main__":
    main()
