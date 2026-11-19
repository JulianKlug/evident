"""Crosswalk families: real phrasings collected during the grading review (2026-11-19), the committed tables'
consistency, and the generated mapping document."""

import csv
import os

import pytest

from evident import grading_tables
from evident.domain import AxisStatus, Category, Certainty, Direction, GradingFamily, MappingVariant, ScalePair
from evident.domain import Strength
from evident.harmonization import harmonize

S, W = Strength.STRONG, Strength.WEAK
F, A = Direction.FOR, Direction.AGAINST
HI, MO, LO, VL = Certainty.HIGH, Certainty.MODERATE, Certainty.LOW, Certainty.VERY_LOW
G, BP, NR = Category.GRADED, Category.BEST_PRACTICE, Category.NO_RECOMMENDATION
P, ALT = MappingVariant.PRIMARY, MappingVariant.ALTERNATIVE
_DOC = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "harmonization_tables.md")

# (family, scales override, raw_strength, raw_certainty, text) → primary (category, strength, direction, certainty),
# alternative (same); None on an axis = unmapped / ungraded
REAL = [
    # ESMO colon cancer 2026, PMC12947638
    (("idsa_esmo", None, "[I, A]", "", "Patients should be informed of the evidence"),
     (G, S, F, HI), (G, S, F, HI)),
    # ESGO vulvar cancer 2023, PMC10359596: B = generally recommended (D1)
    (("idsa_esmo", None, "[V, B]", "", "Patients should be carefully counseled"),
     (G, W, F, VL), (G, S, F, VL)),
    # IDSA/ESMO C = optional → conditional, direction from text (D3)
    (("idsa_esmo", None, "C", "III", "X should not be used routinely"), (G, W, A, LO), (NR, None, None, None)),
    # EGS 4th edition, PMC5583689: roman I = numeric GRADE 1
    (("grade_numeric", None, "[I,D]", "", "Treatment must be individualised"), (G, S, F, VL), (G, S, F, VL)),
    # SMFM cancer in pregnancy, PMC13344446: 3-level letters, C = low or very low (D14)
    (("grade_numeric", ScalePair("S_GRADE_NUM", "E_GRADE_LETTER3"), "GRADE 1C", "",
      "we recommend that surgery not be delayed"), (G, S, F, LO), (G, S, F, VL)),
    # BSH MPN 2019, PMC6519221: 4-level letters
    (("grade_numeric", None, "GRADE 2C", "", "Cardiovascular risk should be assessed"),
     (G, W, F, LO), (G, W, F, LO)),
    # UIP coil embolization 2020, PMC7270724: the grade wins over "we recommend"
    (("grade_numeric", ScalePair("S_GRADE_NUM", "E_GRADE_LETTER3"), "Grade 2C against", "",
      "we recommend against the use of such approaches"), (G, W, A, LO), (G, W, A, VL)),
    # WFSBP OCD/PTSD v3 2023: negative evidence keeps its certainty, direction from the grade
    (("other", ScalePair("S_WFSBP", "E_WFSBP"), "LoE A-/RG 1-", "", "debriefing did not differ"),
     (G, S, A, HI), (G, S, A, HI)),
    (("other", ScalePair("S_WFSBP", "E_WFSBP"), "LoE B/RG 2", "", "Venlafaxine was as effective"),
     (G, W, F, MO), (G, S, F, MO)),
    # EASL-EASD-EASO MASLD 2024, PMC11299976: consensus is never strength (D8)
    (("easl", None, "LoE 3, open recommendation, strong consensus", "", "AFP can be combined with ultrasound"),
     (G, W, F, LO), (NR, None, None, None)),
    # ESPEN home enteral nutrition: AWMF-style grade 0 with SIGN levels
    (("awmf", ScalePair("S_AWMF", "E_SIGN"), "Grade of recommendation 0 – Consensus (90% agreement)", "",
      "HEN can be delivered through a nasal feeding tube"), (G, W, F, None), (NR, None, None, None)),
    # EDF extracorporeal photopheresis 2021, PMC7821314
    (("oxford", ScalePair("S_LETTER_GRADE", "E_OXFORD2009"),
      "Grade of evidence 2b, Strength of recommendation B", "", "ECP should be used"),
     (G, W, F, LO), (G, S, F, MO)),
    # EULAR/PReS Still's disease 2024, PMC11672000: "LoE, strength, agreement, LoA"
    (("oxford", ScalePair("S_LETTER_GRADE", "E_OXFORD2009"), "2a B 100% 9.7", "",
      "sJIA and AOSD are the same disease"), (G, W, F, LO), (G, S, F, MO)),
    # EACTS/AEPC HLHS 2020
    (("esc_class", None, "Class I, Level B", "", "Prostaglandin E1 should be administered"),
     (G, S, F, MO), (G, S, F, LO)),
    # ESC III: against from the grade (D4); C mostly expert consensus (D13)
    (("esc_class", None, "III", "C", "X should be avoided"), (G, S, A, VL), (G, W, A, LO)),
    # ACC/AHA 2015+: B-NR → low (D2)
    (("acc_aha", None, "2a", "B-NR", "It is reasonable to give X"), (G, W, F, LO), (G, S, F, MO)),
    # IUGA OASI 2026, PMC13486146: good practice point
    (("sign", None, "Grade GPP", "", "Every woman should be offered a rectal examination"),
     (BP, None, None, None), (BP, None, None, None)),
    (("sign", None, "Grade B", "2+", "Clinicians should consider a second clinician"),
     (G, W, F, LO), (G, S, F, LO)),
    # NANETS rectal NET: GRADE certainty only, strength not reported (D9)
    (("other", ScalePair("S_NONE", "E_GRADE4"), "", "(grade: LOW)", "T1 tumors should be staged with MRI"),
     (G, None, None, LO), (G, None, None, LO)),
    # EFCNI newborn standards: category A carries GRADE levels
    (("other", ScalePair("S_NONE", "E_EFCNI"), "", "Evidence category A, moderate quality", "X"),
     (G, None, None, MO), (G, None, None, MO)),
    # EPA cognitive impairment 2022: only the letter is printed; certainty derived from it (decision of 2026-11-19)
    (("other", ScalePair("S_LETTER_GRADE", "E_EPA"), "B", "", "No clear superiority of a single antipsychotic"),
     (G, W, F, MO), (G, S, F, MO)),
    (("other", ScalePair("S_LETTER_GRADE", "E_EPA"), "Grade A", "", "Second-generation antipsychotics are recommended"),
     (G, S, F, HI), (G, S, F, MO)),
    # Oxford/SIGN letter without a printed level: certainty stays unmapped (D9)
    (("sign", None, "Grade B", "", "Clinicians should consider a second clinician"), (G, W, F, None), (G, S, F, None)),
    # Consensus-only society: every statement ungraded
    (("consensus_only", None, "", "", "We suggest X"), (BP, None, None, None), (BP, None, None, None)),
    # NCCN categories are not a strength (D7); category 1 still carries certainty
    (("nccn", None, "Preferred", "Category 1", "Give X"), (G, None, None, HI), (G, None, None, HI)),
    # Unknown code: axis unmapped, never guessed
    (("esc_class", None, "Class IV", "Level Z", "Give X"), (G, None, None, None), (G, None, None, None)),
]


def _harmonize(args, variant):
    family, scales, raw_strength, raw_certainty, text = args
    return harmonize(raw_strength, raw_certainty, text, G, GradingFamily(family), scales=scales, variant=variant)


def _assert_invariant(h):
    assert (h.strength_status == AxisStatus.MAPPED) == (h.strength is not None)
    assert (h.certainty_status == AxisStatus.MAPPED) == (h.certainty is not None)


@pytest.mark.parametrize("args, primary, alternative", REAL, ids=[f"{a[0]}:{a[2]}|{a[3]}" for a, _, _ in REAL])
def test_real_phrasing(args, primary, alternative):
    for variant, expected in ((P, primary), (ALT, alternative)):
        h = _harmonize(args, variant)
        assert (h.category, h.strength, h.direction, h.certainty) == expected, variant
        _assert_invariant(h)


def test_grade_family_ignores_variant():
    # GRADE keeps its own rules; the crosswalk alternatives do not apply
    for variant in MappingVariant:
        h = harmonize("2", "C", "We suggest X", G, GradingFamily.GRADE, variant=variant)
        assert (h.strength, h.certainty) == (W, LO)


def test_every_family_has_scales_except_other():
    for family in GradingFamily:
        if family in (GradingFamily.GRADE, GradingFamily.OTHER):
            continue
        assert grading_tables.resolve_scales(family, None) is not None, family


def test_manifest_scales_override_family_default():
    pair = ScalePair("S_GRADE_NUM", "E_GRADE_LETTER3")
    assert grading_tables.resolve_scales(GradingFamily.GRADE_NUMERIC, pair) == pair


def _crosswalk():
    with open(os.path.join(grading_tables.DATA_DIR, "crosswalk.csv"), newline="") as f:
        return list(csv.DictReader(f))


def _codes_from_strength():
    with open(os.path.join(grading_tables.DATA_DIR, "scales.csv"), newline="") as f:
        return {r["scale_key"] for r in csv.DictReader(f) if r["codes_from"] == "strength_grade"}


_LETTER_SCALES = _codes_from_strength()


def _scale_pair(scale_key, axis):
    # Any valid scale on the other axis; the wildcard S_NONE / E_NONE never interferes with a lookup.
    # An evidence scale read off the strength letter (EPA) needs the letter grade scale beside it.
    if scale_key in _LETTER_SCALES:
        return ScalePair("S_LETTER_GRADE", scale_key)
    return ScalePair(scale_key, "E_NONE") if axis == "strength" else ScalePair("S_NONE", scale_key)


def _labels_in(row, label):
    """(raw_strength, raw_certainty) carrying the native code where the guideline prints it."""
    if row["axis"] == "strength" or row["scale_key"] in _LETTER_SCALES:
        return label, ""
    return "", label


ROUND_TRIP = [r for r in _crosswalk() if r["native_code"] != grading_tables.WILDCARD]


@pytest.mark.parametrize("row", ROUND_TRIP, ids=[f"{r['scale_key']}:{r['native_code']}" for r in ROUND_TRIP])
def test_every_native_code_maps_to_its_row(row):
    pair = _scale_pair(row["scale_key"], row["axis"])
    labels = [row["native_code"]] + [a for a in row["aliases"].split("|") if a]
    for label in labels:
        strength, evidence = grading_tables.match(pair, *_labels_in(row, label))
        found = strength if row["axis"] == "strength" else evidence
        assert found is not None, label
        assert found.value.value == row["harmonised"], label


def test_mapping_document_is_up_to_date():
    with open(_DOC) as f:
        assert f.read() == grading_tables.render_doc(), "run: python scripts/export_grading_tables.py --doc"
