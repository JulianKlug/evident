"""Tests for evident.publishing: files written, determinism, captions, self-contained HTML, safe swap."""

import csv
import json
import os
import re

import pytest

from evident import publishing
from evident.analytics import (
    Edition,
    GuidelineFacts,
    HarmonizedRow,
    caption_facts,
    certainty_distribution,
    guideline_summary,
    headline,
    strength_by_certainty,
    trends,
)
from evident.domain import AxisStatus, Category, Certainty, Direction, GateResult, HarmonizedGrade, Strength
from evident.publishing import PublishBundle, ValidationSummary, render

_M, _UG = AxisStatus.MAPPED, AxisStatus.UNGRADED

_EXPECTED = [
    "dashboard/index.html",
    "figures/captions.md",
    "figures/fig1_certainty_by_society.png", "figures/fig1_certainty_by_society.svg",
    "figures/fig2_strong_by_society.png", "figures/fig2_strong_by_society.svg",
    "figures/fig3_strength_by_certainty.png", "figures/fig3_strength_by_certainty.svg",
    "figures/fig4_trends.png", "figures/fig4_trends.svg",
    "tables/certainty.csv", "tables/guidelines.csv", "tables/headline.csv", "tables/recommendations.csv",
    "tables/strength_by_certainty.csv", "tables/trends.csv",
]


def _rec(doi, society, year, strength, certainty, text="We recommend X", edition=Edition.CURRENT):
    grade = HarmonizedGrade(Category.GRADED, strength, Direction.FOR, certainty, _M, _M)
    return HarmonizedRow(doi, society, year, grade, edition, text, 3, str(strength), str(certainty))


_ROWS = [
    _rec("10.1/a", "ACP", 2023, Strength.STRONG, Certainty.HIGH, text="Use </script><b>x</b> wisely"),
    _rec("10.1/a", "ACP", 2023, Strength.WEAK, Certainty.LOW),
    _rec("10.1/old", "ACP", 2018, Strength.WEAK, Certainty.VERY_LOW, edition=Edition.SUPERSEDED),
    _rec("10.1/b", "SCCM", 2021, Strength.STRONG, Certainty.MODERATE),
    HarmonizedRow("10.1/b", "SCCM", 2021, HarmonizedGrade(Category.BEST_PRACTICE, None, None, None, _UG, _UG),
                  Edition.CURRENT, "Best practice: do Y", None, "BPS", ""),
]
_FACTS = [
    GuidelineFacts("10.1/a", "ACP", 2023, "Guideline A", "topic-a", Edition.CURRENT),
    GuidelineFacts("10.1/old", "ACP", 2018, "Guideline A old", "topic-a", Edition.SUPERSEDED),
    GuidelineFacts("10.1/b", "SCCM", 2021, "Guideline B", "topic-b", Edition.CURRENT, joint_with=("ESICM",)),
]
_VALIDATION = ValidationSummary(GateResult.PASS, 12, 5, 0.929, (0.867, 0.978), 0.868, 1.0, 0.935, (0.9, 0.97))


def _bundle(labelled=frozenset({"10.1/a"}), validation=_VALIDATION, rows=None):
    rows = _ROWS if rows is None else rows
    return PublishBundle(
        snapshot_id=7, extractor_version_id="113889035d33abcdef", model="qwen38-64k:latest@12af24e3ee72",
        code_sha="abcdef0123456", validation=validation,
        headline=headline(rows, 7, "113889035d33abcdef", "m"),
        certainty=certainty_distribution(rows), crosstab=strength_by_certainty(rows), trends=trends(rows),
        guidelines=guideline_summary(rows, _FACTS), captions=caption_facts(rows, _FACTS, set(labelled)),
        recs=rows,
    )


def _files(out_dir):
    return sorted(os.path.relpath(os.path.join(root, f), out_dir).replace(os.sep, "/")
                  for root, _, files in os.walk(out_dir) for f in files)


def _read(path, mode="r"):
    with open(path, mode) as f:
        return f.read()


def test_writes_every_output(tmp_path):
    out = str(tmp_path / "out")
    written = render(_bundle(), out)
    assert _files(out) == _EXPECTED
    assert sorted(os.path.relpath(p, out) for p in written) == _EXPECTED


def test_keeps_other_files_in_out_dir(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "evident.sqlite").write_text("db")
    render(_bundle(), str(out))
    assert (out / "evident.sqlite").read_text() == "db"


def test_byte_identical_across_renders(tmp_path):
    first, second = str(tmp_path / "1"), str(tmp_path / "2")
    render(_bundle(), first)
    render(_bundle(), second)
    for rel in _EXPECTED:
        assert _read(os.path.join(first, rel), "rb") == _read(os.path.join(second, rel), "rb"), rel


def test_tables(tmp_path):
    out = str(tmp_path)
    render(_bundle(), out)
    headline_rows = list(csv.DictReader(open(os.path.join(out, "tables", "headline.csv"))))
    assert [r["society"] for r in headline_rows] == ["ACP", "SCCM", "ALL"]
    assert headline_rows[1]["pct_high_moderate"] == "100.0"

    recs = list(csv.DictReader(open(os.path.join(out, "tables", "recommendations.csv"))))
    assert len(recs) == len(_ROWS)
    assert recs[-1]["page"] == "" and recs[-1]["category"] == "best_practice"

    guidelines = list(csv.DictReader(open(os.path.join(out, "tables", "guidelines.csv"))))
    assert {g["doi"]: g["joint_with"] for g in guidelines}["10.1/b"] == "ESICM"


def test_captions_state_counts_and_flags(tmp_path):
    render(_bundle(), str(tmp_path))
    captions = _read(tmp_path / "figures" / "captions.md")
    assert "Current editions of 2 ACP and SCCM guidelines (2018–2023): 4 recommendations, 3 graded." in captions
    assert "SCCM: extraction accuracy unvalidated (no labelled SCCM guideline)." in captions
    assert "10.1/b (SCCM with ESICM)" in captions
    assert "No guideline excluded." in captions
    assert "Snapshot 7, extractor version 113889035d33, model qwen38-64k:latest@12af24e3ee72." in captions
    assert "graded F1 0.929 [0.867, 0.978]" in captions
    assert "1 superseded edition(s)" in captions


def test_no_unvalidated_sentence_when_all_labelled(tmp_path):
    render(_bundle(labelled={"10.1/a", "10.1/b"}), str(tmp_path))
    assert "unvalidated" not in _read(tmp_path / "figures" / "captions.md")
    assert "unvalidated" not in _read(tmp_path / "dashboard" / "index.html")


def test_dashboard_is_self_contained(tmp_path):
    render(_bundle(), str(tmp_path))
    html = _read(tmp_path / "dashboard" / "index.html")

    data = json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S).group(1))
    assert len(data["recs"]) == len(_ROWS)
    assert data["recs"][0]["text"] == "Use </script><b>x</b> wisely"
    assert "</script><b>" not in html  # embedded text can't close the script tag

    urls = re.findall(r"https?://[^\s\"'<>]+", html)
    assert all(u.startswith("https://doi.org/") for u in urls), urls
    assert "SCCM: extraction accuracy unvalidated" in html
    assert html.count("data:image/svg+xml;base64,") == 4


def test_dashboard_without_validation(tmp_path):
    render(_bundle(validation=None), str(tmp_path))
    assert "Not validated." in _read(tmp_path / "figures" / "captions.md")


def test_empty_snapshot_renders(tmp_path):
    render(_bundle(rows=[]), str(tmp_path))
    assert _files(str(tmp_path)) == _EXPECTED


def test_failure_keeps_previous_outputs(tmp_path, monkeypatch):
    out = str(tmp_path)
    render(_bundle(), out)
    before = {rel: _read(os.path.join(out, rel), "rb") for rel in _EXPECTED}

    def broken(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(publishing, "_write_dashboard", broken)
    with pytest.raises(RuntimeError):
        render(_bundle(labelled={"10.1/a", "10.1/b"}), out)

    assert _files(out) == _EXPECTED  # no staging dir left behind
    assert {rel: _read(os.path.join(out, rel), "rb") for rel in _EXPECTED} == before
