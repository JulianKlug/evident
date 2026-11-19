"""Crosswalk families: real phrasings collected during the grading review (2026-11-19), the committed tables'
consistency, and the generated mapping document."""

import csv
import os

import pytest

from evident import grading_tables
from evident.domain import AxisStatus, Category, Certainty, Direction, GradingFamily, MappingVariant, ScalePair
from evident.domain import Strength
from evident.harmonization import harmonize
from tests.conftest import needs_local_data

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
    (("esmo", None, "[I, A]", "", "Patients should be informed of the evidence"),
     (G, S, F, HI), (G, S, F, HI)),
    # ESGO vulvar cancer 2023, PMC10359596: B = generally recommended (D1)
    (("esmo", None, "[V, B]", "", "Patients should be carefully counseled"),
     (G, W, F, VL), (G, S, F, VL)),
    # IDSA/ESMO C = optional → conditional, direction from text (D3)
    (("esmo", None, "C", "III", "X should not be used routinely"), (G, W, A, LO), (NR, None, None, None)),
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
        # The row itself, not just one with the same primary value
        assert found.native_code == row["native_code"], label
        assert found.value.value == row["harmonised"], label
        assert (found.alternative.value if found.alternative else "") == row["alternative"], label
        assert (found.direction.value.upper() if found.direction else "") == (
            row["direction"] if row["direction"] in ("FOR", "AGAINST") else ""), label


def test_mapping_document_is_up_to_date():
    with open(_DOC) as f:
        assert f.read() == grading_tables.render_doc(), "run: python scripts/export_grading_tables.py --doc"


# ---------------------------------------------------------------- review fixes (PR #5)

# Negative wording → AGAINST; "cannot be recommended" with an evidence-gap clause → no recommendation
NEGATIVE_TEXT = [
    ("X is not recommended", G, A),
    ("Routine use of X is not recommended in children", G, A),
    ("Do not offer X to people with Y", G, A),
    ("X is not indicated after surgery", G, A),
    ("Avoid X in pregnancy", G, A),
    ("X should be avoided", G, A),
    ("X cannot be recommended for routine use", G, A),
    ("X sollte nicht angewendet werden", G, A),
    ("X soll bei Kindern nicht eingesetzt werden", G, A),
    ("X wird nicht empfohlen", G, A),
    ("X sollte nicht nur bei Erwachsenen eingesetzt werden", G, F),   # "nicht nur" = not only
    ("To avoid infection, give X", G, F),
    ("In adults with MASLD, nutraceuticals cannot be recommended since there is insufficient evidence", NR, None),
]


@pytest.mark.parametrize("text, category, direction", NEGATIVE_TEXT, ids=[t for t, _, _ in NEGATIVE_TEXT])
@pytest.mark.parametrize("family, raw_strength, raw_certainty", [
    (GradingFamily.OXFORD, "B", "2"),          # letter grade: direction as stated
    (GradingFamily.GRADE, "Strong", "High"),   # GRADE path gets the same rules
])
def test_negative_wording(family, raw_strength, raw_certainty, text, category, direction):
    h = harmonize(raw_strength, raw_certainty, text, G, family)
    assert (h.category, h.direction) == (category, direction)


# Combined and legacy notations (spec §6) → (strength, certainty); ambiguous parts stay unmapped
NOTATION = [
    ("esc_class", "I-A", "", S, HI),
    ("esc_class", "IIa-B", "", W, MO),
    ("esc_class", "Class 2a", "B", W, MO),             # ESC numeric alias
    ("esc_class", "Class II a", "", W, None),          # "a" is part of the class, not level A
    ("esc_class", "ILA", "B", W, MO),                  # OCR: L misread for I (NI9RV3E7)
    ("esc_class", "LIB", "C", W, VL),
    ("esc_class", "ILI", "C", S, VL),
    ("acc_aha", "2a-B-NR", "", W, LO),
    ("acc_aha", "COR 2a, LOE B-NR", "", W, LO),
    ("idsa", "A", "II", S, LO),                        # classic IDSA: evidence I–III
    ("idsa", "AII", "", S, LO),                        # IDSA letter+roman in one token
    ("idsa", "A-II", "", S, LO),
    ("esmo", "[II, B]", "", W, MO),                    # ESMO: evidence I–V
]


@pytest.mark.parametrize("family, raw_strength, raw_certainty, strength, certainty", NOTATION,
                         ids=[f"{n[0]}:{n[1]}|{n[2]}" for n in NOTATION])
def test_notation(family, raw_strength, raw_certainty, strength, certainty):
    h = harmonize(raw_strength, raw_certainty, "Give X", G, GradingFamily(family))
    assert (h.strength, h.certainty) == (strength, certainty)


# ---------------------------------------------------------------- legacy ERS ground truth (M6.5a pneumology)

def _ers(key):
    from extraction.datasets import load_ers_datasets
    return next(ds for ds in load_ers_datasets() if ds.key == key)


_ESC_STRENGTH = {"I": (S, F), "IIa": (W, F), "IIb": (W, F), "III": (S, A)}
_ESC_LEVEL = {"A": HI, "B": MO, "C": VL}


@needs_local_data
def test_ers_esc_guideline_maps_every_row():
    # NI9RV3E7: ESC/ERS pulmonary hypertension 2015, Class I–III + Level A–C (legacy ESC_ERS)
    rows = _ers("NI9RV3E7").ground_truth_df
    assert len(rows) == 217
    for _, r in rows.iterrows():
        h = harmonize(r["class"], r["LOE"], r["recommendation"], G, GradingFamily.ESC_CLASS)
        assert (h.strength, h.direction) == _ESC_STRENGTH[r["class"]], (r["class"], r["recommendation"][:60])
        assert h.certainty == _ESC_LEVEL[r["LOE"]]


_ESCMID_STRENGTH = {"A": S, "B": W, "C": W, "D": W}
_ESCMID_QUALITY = {"1": HI, "2": LO, "3": VL}


@needs_local_data
def test_ers_escmid_guideline_maps_every_row():
    # BDYDTUHA: ESCMID/ERS aspergillosis 2015, SoR A–D (D = against) + QoE I–III (legacy ABCD_123)
    rows = _ers("BDYDTUHA").ground_truth_df
    assert len(rows) == 60
    for _, r in rows.iterrows():
        h = harmonize(r["class"], r["LOE"], r["recommendation"], G, GradingFamily.IDSA)
        assert h.certainty == _ESCMID_QUALITY[str(r["LOE"])], r["recommendation"][:60]
        if not r["class"]:
            assert h.strength_status == AxisStatus.UNMAPPED  # quality-only statement
            continue
        assert h.strength == _ESCMID_STRENGTH[r["class"]], (r["class"], r["recommendation"][:60])
        if r["class"] == "D":
            assert h.direction == A


def test_codes_sharing_a_label_must_agree_on_the_whole_mapping(tmp_path, monkeypatch):
    # "B" twice in one scale, same primary value but different sensitivity alternative → refused
    for name in ("scales.csv", "families.csv", "decisions.csv"):
        (tmp_path / name).write_text(open(os.path.join(grading_tables.DATA_DIR, name)).read())
    rows = _crosswalk() + [dict(_crosswalk()[0], native_code="Conditional", harmonised="CONDITIONAL",
                                alternative="STRONG")]
    with open(tmp_path / "crosswalk.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    monkeypatch.setattr(grading_tables, "DATA_DIR", str(tmp_path))
    grading_tables._tables.cache_clear()
    try:
        with pytest.raises(grading_tables.GradingTableError, match="maps to both"):
            grading_tables._tables()
    finally:
        monkeypatch.undo()
        grading_tables._tables.cache_clear()


# ---------------------------------------------------------------- survey notations (review of PR #5, round 2)

def _survey_scales(society) -> ScalePair:
    if isinstance(society, ScalePair):  # a historical pair no surveyed society prints
        return society
    with open(grading_tables.SURVEY_PATH, encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["society"] == society)
    return ScalePair(row["strength_scale"], row["evidence_scale"])


# (society, raw_strength, raw_certainty, text) → (category, strength, direction, certainty) from the survey's
# notation examples: consensus agreement keeps the evidence grade (D8, D9); ACC/AHA roman classes
SURVEY = [
    ("ESPNIC", "strong agreement", "quality of evidence D", "Lung ultrasound should be used", (G, None, None, VL)),
    ("ESNM", "Statement endorsed, overall agreement 95%", "GRADE B", "PPIs are effective", (G, None, None, MO)),
    ("ESSR", "strong consensus (100% agreement)", "Level of evidence 1", "Injection is safe", (G, None, None, HI)),
    ("ESNR", "broad consensus", "Level of Evidence (LoE): 2", "Facet injection is effective", (G, None, None, MO)),
    ("ESPE", "A +++", "", "Growth should be monitored", (BP, None, None, None)),
    ("SOAP", "COR IIa", "LOE C-LD", "Neuraxial anesthesia is reasonable", (G, W, F, VL)),
    ("ACC", "Class I", "Level A", "Statins are recommended", (G, S, F, HI)),
    # 2016 AHA/ACC PAD (PMC5477786) and 2023 HRS/APHRS/LAHRS pacing (PMC11062890) print 2015+ LOE
    ("AHA", "I", "B-NR", "Patients at increased risk of PAD should undergo a history", (G, S, F, LO)),
    ("ACC", "III: Harm", "B-R", "CTA should not be performed in asymptomatic PAD", (G, S, A, MO)),
    ("HRS", "2a", "B-NR", "Periodic assessment of ventricular function is reasonable", (G, W, F, LO)),
    ("APHRS", "3: No Benefit", "B-R", "CRT is not useful", (G, W, A, MO)),
    ("NLA", "III - Harm", "B-R", "Niacin should not be used", (G, S, A, MO)),
    ("NLA", "III: No Benefit", "B-R", "Lp(a) apheresis is not useful", (G, W, A, MO)),
    # Pre-2015 ACC/AHA: I/IIa/IIb/undivided III + A/B/C, graded as ESC (D4: III strong against)
    (ScalePair("S_ESC_CLASS", "E_ESC"), "Class III", "Level C", "Routine screening is not recommended", (G, S, A, VL)),
]


@pytest.mark.parametrize("society, raw_strength, raw_certainty, text, expected", SURVEY,
                         ids=[f"{getattr(s, 'evidence', s)}:{rs}|{rc}" for s, rs, rc, _, _ in SURVEY])
def test_survey_notation(society, raw_strength, raw_certainty, text, expected):
    h = harmonize(raw_strength, raw_certainty, text, G, GradingFamily.OTHER, scales=_survey_scales(society))
    assert (h.category, h.strength, h.direction, h.certainty) == expected
    _assert_invariant(h)


_ACCAHA = ScalePair("S_ACCAHA_COR", "E_ACCAHA")

# 2015+ Class III notations → (primary, alternative) (strength, direction); a bare III is ambiguous there
ACCAHA_CLASS_III = [
    ("III: Harm", (S, A), (S, A)),
    ("III - Harm", (S, A), (S, A)),
    ("3: Harm", (S, A), (S, A)),
    ("III: No Benefit", (W, A), (S, A)),
    ("III - No Benefit", (W, A), (S, A)),
    ("3 – No benefit", (W, A), (S, A)),
    ("Class III", (None, None), (None, None)),
    ("3", (None, None), (None, None)),
]


@pytest.mark.parametrize("raw_strength, primary, alternative", ACCAHA_CLASS_III,
                         ids=[s for s, _, _ in ACCAHA_CLASS_III])
def test_accaha_class_iii(raw_strength, primary, alternative):
    for variant, expected in ((P, primary), (ALT, alternative)):
        h = harmonize(raw_strength, "B-R", "X is not useful", G, GradingFamily.OTHER, scales=_ACCAHA, variant=variant)
        assert (h.strength, h.direction) == expected, variant
        _assert_invariant(h)


# ---------------------------------------------------------------- coverage and reported numbers

_METHODS = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "methods_grading_harmonisation.md")
_MIN_COVERAGE = 0.73  # spec §8 exit


def test_survey_coverage_meets_the_spec_exit():
    coverage = grading_tables.survey_coverage()
    assert coverage.share >= _MIN_COVERAGE
    # Every graded society not covered says why (D7: not a recommendation grade)
    assert all(r["harmonisability"] for r in coverage.unsupported)


def test_methods_text_states_the_committed_numbers():
    # The methods draft is prose; its numbers must match the committed tables
    with open(os.path.join(grading_tables.DATA_DIR, "scales.csv"), newline="") as f:
        scales = list(csv.DictReader(f))
    codes = _crosswalk()
    with_alternative = [r for r in codes if r["alternative"]]
    coverage = grading_tables.survey_coverage()
    included = coverage.n_surveyed - len(coverage.excluded)
    with open(_METHODS) as f:
        text = f.read()

    n_strength = sum(r["axis"] == "strength" for r in scales)
    expected = [
        f"{len(scales)} native scales: {n_strength} for strength and {len(scales) - n_strength} for evidence",
        f"({len(codes)} native codes)",
        f"({len(with_alternative)} codes; {100 * len(with_alternative) / len(codes):.1f}%)",
        f"{len(coverage.covered)} of the {len(coverage.graded)} graded societies ({100 * coverage.share:.1f}%)",
        f"{len(coverage.both_axes)} ({100 * len(coverage.both_axes) / len(coverage.graded):.1f}%) on both axes",
        f"{included} of the {coverage.n_surveyed} societies ({100 * included / coverage.n_surveyed:.1f}%)",
    ]
    for phrase in expected:
        assert phrase in text, phrase


def test_tables_hash_changes_with_any_table(tmp_path, monkeypatch):
    for name in os.listdir(grading_tables.DATA_DIR):
        (tmp_path / name).write_bytes(open(os.path.join(grading_tables.DATA_DIR, name), "rb").read())
    monkeypatch.setattr(grading_tables, "DATA_DIR", str(tmp_path))
    before = grading_tables.tables_sha256()
    with open(tmp_path / "decisions.csv", "a") as f:
        f.write("D99,q,r,a,why\n")
    assert grading_tables.tables_sha256() != before
