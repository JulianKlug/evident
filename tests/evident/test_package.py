"""publish_package on fakes: read-only frozen source, fail-closed audit, every export, byte-identical reruns.

    World (4 guidelines, SCCM 10.1000/e held out) ─snapshot─validate─publish─▶ freeze (DELETE journal)
        ─▶ Store READ_ONLY ─▶ publish_package ─▶ <tmp>/paper/ + <tmp>/out/
"""

import csv
import dataclasses
import json
import os
import sqlite3

import pytest

from evident import publishing
from evident.corpus import file_sha256
from evident.dedup import DedupVerdictError, Verdict, write_adjudications
from evident.domain import SnapshotState
from evident.paper import AnalysisConfig
from evident.pipeline import (
    PackageAuditError,
    PackageInputs,
    SnapshotStart,
    TreeState,
    ValidationMissingError,
    duplicate_candidates,
    publish,
    publish_package,
    snapshot,
    validate,
)
from evident.publishing import DraftFieldError
from evident.store import AccessMode, Store
from evident.validation import HeldoutLabelError
from tests.evident.pipeline_fakes import CONFIG, GUIDELINES, World, write_heldout_xlsx
from tests.legacy_golden import FakeEncoder

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TEMPLATE = os.path.join(_REPO, "tests", "fixtures", "paper", "draft.md.j2")  # the real draft is not tracked
_DOIS = ("10.1000/a", "10.1000/b", "10.1000/c", "10.1000/e")
_HELDOUT = "10.1000/e"
_SPOT = "10.1000/b"
_QUERIES = (("ACP", "pubmed", "q1"), ("ESICM", "pubmed", "q2"), ("SCCM", "pubmed", "q3"))

_EXPECTED = [
    "analysis_manifest.json", "captions.md", "corpus_flow.csv", "draft.md", "duplicates.csv", "excluded_guidelines.csv",
    "figure_data.csv", "methods_facts.json", "raw_harmonized_mapping.csv", "spotcheck_bias.csv",
    "table1_society.csv", "table2_certainty_strength.csv", "table_s1_guideline_mean.csv", "table_s2_without_dedup.csv",
    "validation_development.json", "validation_sccm_heldout.json",
]


class Frozen:
    """A published World snapshot, frozen to a read-only DB copy, with every package input on disk."""

    def __init__(self, tmp_path, exclude=None):
        self.world = World(tmp_path, dois=_DOIS)
        deps = self.world.deps()
        snapshot(deps, SnapshotStart.NEW, CONFIG, [exclude] if exclude else [])
        validate(deps)
        publish(deps)
        self.world.store.close()

        self.db = os.path.join(self.world.root, "db.sqlite")
        conn = sqlite3.connect(self.db)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode = DELETE")
        conn.close()
        self.store = Store.open(self.db, AccessMode.READ_ONLY)

        self.root = self.world.root
        self.candidates = self._write_candidates()
        self.spotcheck = self._write_spotcheck()
        self.heldout = os.path.join(self.root, "heldout")
        texts = GUIDELINES[_HELDOUT][3]
        write_heldout_xlsx(self.heldout, _HELDOUT, [
            (texts[0][0], "strong recommendation, for", "moderate certainty evidence", 2),
            (texts[1][0], "conditional recommendation, against", "very low certainty evidence", 2),
            ("Best practice: involve the family", "Best practice statement", "NA", 3),
        ])
        self.package = os.path.join(self.root, "paper")
        self.config_path = os.path.join(self.root, "paper.json")
        self.revisions = []  # embedding revisions the package asked for
        self.write_config()

    def write_config(self, **overrides):
        values = dict(
            name="paper_test", snapshot_id=1, search_start="2016-01-01", search_cutoff="2026-10-04",
            db_sha256=file_sha256(self.db), manifest_sha256=file_sha256(self.world.manifest),
            candidates_sha256=file_sha256(self.candidates), candidates_path=self.candidates,
            n_guidelines=len(_DOIS), n_current=len(_DOIS), bootstrap_n=200, bootstrap_seed=0,
            heldout_dois=[_HELDOUT], spotcheck_dir=self.spotcheck,
            focused_update_overlap_max=0,
            heldout_labels_sha256={n: file_sha256(os.path.join(self.heldout, n)) for n in os.listdir(self.heldout)},
        )
        values.update(overrides)
        with open(self.config_path, "w") as f:
            json.dump(values, f)

    def inputs(self, **overrides) -> PackageInputs:
        with open(self.config_path) as f:
            config = AnalysisConfig.from_json(f.read())
        inputs = PackageInputs(self.db, self.package, config, self.config_path, self.heldout, _QUERIES,
                               TreeState.CLEAN, self._load_similarity, _TEMPLATE)
        return dataclasses.replace(inputs, **overrides)

    def reopen_writable(self):
        """Close the read-only store; return a writable connection. Call refreeze() after the edit."""
        self.store.close()
        return sqlite3.connect(self.db)

    def refreeze(self, conn):
        conn.commit()
        conn.close()
        self.store = Store.open(self.db, AccessMode.READ_ONLY)
        self.write_config()

    def _load_similarity(self, revision: str):
        self.revisions.append(revision)
        return FakeEncoder()

    def deps(self):
        return dataclasses.replace(self.world.deps(), store=self.store)

    def run(self, **overrides):
        return publish_package(self.deps(), 1, self.inputs(**overrides))

    def _write_candidates(self) -> str:
        path = os.path.join(self.root, "candidates.csv")
        rows = [f"{d},{GUIDELINES[d][0]},{GUIDELINES[d][1]},T,J,1,pubmed,include,," for d in _DOIS]
        rows.append("10.1000/x,ACP,2020,Review,J,2,pubmed_broad,exclude,not_guideline,")
        with open(path, "w") as f:
            f.write("doi,society,year,title,journal,pmid,source,decision,reason,duplicate_of\n" + "\n".join(rows) + "\n")
        return path

    def _write_spotcheck(self, matched=(0, 1)) -> str:
        directory = os.path.join(self.root, "spotcheck")
        os.makedirs(directory, exist_ok=True)
        reference = [{"category": "graded", "strength": "weak", "direction": "for", "certainty": "low"},
                     {"category": "graded", "strength": "strong", "direction": "against", "certainty": "high"},
                     {"category": "best_practice"}]
        data = {"doi": _SPOT, "matches": [{"extracted_id": i} for i in matched], "reference": reference}
        with open(os.path.join(directory, "10.1000_b.spotcheck.json"), "w") as f:
            json.dump(data, f)
        return directory


@pytest.fixture
def frozen(tmp_path):
    f = Frozen(tmp_path)
    yield f
    f.store.close()


def _files(directory):
    return sorted(os.listdir(directory))


def _read(path, mode="r"):
    with open(path, mode) as f:
        return f.read()


def _csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ── outputs ─────────────────────────────────────────────────────

def test_writes_every_export_and_standard_outputs(frozen):
    report = frozen.run()

    assert _files(frozen.package) == _EXPECTED
    assert sorted(os.path.basename(p) for p in report.package_paths) == _EXPECTED
    assert os.path.isfile(os.path.join(frozen.world.publish_dir, "dashboard", "index.html"))
    assert report.heldout.n_guidelines == 1


def test_frozen_db_never_written(frozen):
    before = file_sha256(frozen.db)
    frozen.run()
    assert file_sha256(frozen.db) == before
    assert not os.path.exists(frozen.db + "-wal")


def test_byte_identical_reruns(frozen):
    frozen.run()
    first = {n: _read(os.path.join(frozen.package, n), "rb") for n in _EXPECTED}
    frozen.run()
    assert {n: _read(os.path.join(frozen.package, n), "rb") for n in _EXPECTED} == first


def test_table1_values_and_na(frozen):
    frozen.run()
    t1 = {r["society"]: r for r in _csv(os.path.join(frozen.package, "table1_society.csv"))}
    assert list(t1) == ["ACP", "ESICM", "SCCM", "ALL"]
    # ACP 10.1000/a: strong against, conditional for, strong for → 2/3 strong, 1/3 against
    assert (t1["ACP"]["strong_num"], t1["ACP"]["strong_den"], t1["ACP"]["strong_pct"]) == ("2", "3", "66.7")
    assert (t1["ACP"]["against_num"], t1["ACP"]["against_pct"]) == ("1", "33.3")
    assert t1["ALL"]["n_graded"] == "9"
    assert t1["ACP"]["strong_ci_low"] == t1["ACP"]["strong_ci_high"] == "66.7"  # one ACP guideline


def test_manifest_hashes_every_other_output(frozen):
    frozen.run()
    manifest = json.loads(_read(os.path.join(frozen.package, "analysis_manifest.json")))

    assert sorted(manifest["outputs"]) == [n for n in _EXPECTED if n != "analysis_manifest.json"]
    for name, sha in manifest["outputs"].items():
        assert file_sha256(os.path.join(frozen.package, name)) == sha
    assert manifest["inputs"]["database"]["sha256"] == file_sha256(frozen.db)
    assert list(manifest["inputs"]["heldout_labels"]) == ["10.1000_e.xlsx"]
    assert manifest["code"] == {"analysis_sha": "sha1", "extraction_shas": ["sha1"]}
    assert manifest["snapshot"]["state"] == "published"
    assert manifest["bootstrap"]["n"] == 200


def test_validation_samples_kept_apart(frozen):
    frozen.run()
    dev = json.loads(_read(os.path.join(frozen.package, "validation_development.json")))
    held = json.loads(_read(os.path.join(frozen.package, "validation_sccm_heldout.json")))

    stored = json.loads(frozen.store.latest_validation(1).report_json)
    assert dev["report"] == stored  # the gate's own report, unchanged
    assert (dev["sample"], dev["in_sample"], dev["gated"]) == ("development", True, True)
    assert (held["sample"], held["in_sample"], held["gated"]) == ("heldout", False, False)
    assert held["dois"] == [_HELDOUT]
    assert (held["report"]["tp"], held["report"]["fp"], held["report"]["fn"]) == (2, 0, 0)
    assert "ignores clustering" in held["report"]["interval_note"]


def test_sccm_caption_changes_only_with_heldout_results(frozen):
    captions = _read(os.path.join(frozen.world.publish_dir, "figures", "captions.md"))
    assert "SCCM: extraction accuracy unvalidated" in captions  # plain publish, before the package

    frozen.run()
    for path in (os.path.join(frozen.world.publish_dir, "figures", "captions.md"),
                 os.path.join(frozen.package, "captions.md")):
        text = _read(path)
        assert "SCCM: evaluated separately on 1 held-out guidelines; not part of the 2-guideline development gate." in text
        assert "SCCM: extraction accuracy unvalidated" not in text
    assert "SCCM: extraction accuracy unvalidated" not in _read(
        os.path.join(frozen.world.publish_dir, "dashboard", "index.html"))  # held-out row instead


def test_package_figures_carry_cis_and_heldout_row(frozen):
    frozen.run()
    captions = _read(os.path.join(frozen.world.publish_dir, "figures", "captions.md"))
    assert "Whiskers: 95% guideline-cluster bootstrap interval." in captions
    html = _read(os.path.join(frozen.world.publish_dir, "dashboard", "index.html"))
    assert "Held-out SCCM, 1 guidelines" in html


def test_corpus_flow_and_spotcheck(frozen):
    frozen.run()
    flow = {(r["stage"], r["item"]): r["n"] for r in _csv(os.path.join(frozen.package, "corpus_flow.csv"))}
    assert (flow["candidates", "all"], flow["included", "all"], flow["excluded", "not_guideline"]) == ("5", "4", "1")
    assert flow["excluded", "pdf_unavailable"] == "0"

    bias = {r["metric"]: r for r in _csv(os.path.join(frozen.package, "spotcheck_bias.csv"))}
    # 10.1000/b published: weak/low, strong/high, weak/very low; reference: weak/low, strong/high (+1 ungraded)
    assert (bias["strong"]["published_num"], bias["strong"]["published_den"]) == ("1", "3")
    assert (bias["strong"]["truth_num"], bias["strong"]["truth_den"]) == ("1", "2")
    assert (bias["strong"]["correct_num"], bias["strong"]["correct_den"]) == ("1", "2")  # rows 0, 1
    assert bias["ungraded_share"]["correct_num"] == "NA"


def test_package_without_draft_template(frozen):
    frozen.run(draft_template=None)

    assert _files(frozen.package) == [n for n in _EXPECTED if n != "draft.md"]
    manifest = json.loads(_read(os.path.join(frozen.package, "analysis_manifest.json")))
    assert manifest["inputs"]["template"] is None


def test_draft_rendered_from_exports(frozen):
    frozen.run()
    draft = _read(os.path.join(frozen.package, "draft.md"))
    t1 = {r["society"]: r for r in _csv(os.path.join(frozen.package, "table1_society.csv"))}
    assert f"{t1['ALL']['strong_pct']}% were strong ({t1['ALL']['strong_num']}/{t1['ALL']['strong_den']}" in draft
    assert "We ran 3 PubMed queries (1 per society" in draft
    assert "{{" not in draft


# ── fail closed ─────────────────────────────────────────────────

@pytest.mark.parametrize("override, message", [
    ({"db_sha256": "0" * 64}, "database"),
    ({"manifest_sha256": "0" * 64}, "manifest"),
    ({"n_current": 3}, "4 current editions, expected 3"),
    ({"n_guidelines": 5}, "4 manifest guidelines, expected 5"),
    ({"snapshot_id": 2}, "config paper_test names 2"),
    ({"heldout_dois": ["10.1000/zz"]}, "held-out 10.1000/zz is not a manifest guideline"),
    ({"candidates_sha256": "0" * 64}, "candidates"),
    ({"spotcheck_dir": "/nope"}, "spot-check dir /nope missing"),
    ({"heldout_labels_sha256": {"10.1000_e.xlsx": "0" * 64}}, "held-out labels"),
    ({"heldout_labels_sha256": {}}, "held-out labels"),
])
def test_audit_refuses_a_different_source(frozen, override, message):
    frozen.write_config(**override)
    with pytest.raises(PackageAuditError) as e:
        frozen.run()
    assert any(message in p for p in e.value.problems), e.value.problems
    assert not os.path.exists(frozen.package)


def test_audit_refuses_a_missing_draft_template(frozen):
    with pytest.raises(PackageAuditError, match="draft template /nope.j2 missing"):
        frozen.run(draft_template="/nope.j2")
    assert not os.path.exists(frozen.package)


def test_audit_refuses_candidates_not_matching_manifest(frozen):
    with open(frozen.candidates, "a") as f:
        f.write("10.1000/zz,ACP,2021,T,J,3,pubmed,include,,\n")
    frozen.write_config()
    with pytest.raises(PackageAuditError, match="included candidate 10.1000/zz has no manifest row"):
        frozen.run()


def test_audit_refuses_excluded_member(tmp_path):
    f = Frozen(tmp_path, exclude=("10.1000/c", "scanned PDF"))
    with pytest.raises(PackageAuditError, match=r"10.1000/c: excluded \(scanned PDF\)"):
        f.run()
    f.store.close()


def test_audit_refuses_pdf_drift(frozen):
    frozen.store.close()
    conn = sqlite3.connect(frozen.db)
    conn.execute("UPDATE guideline SET pdf_sha256 = 'other' WHERE doi = '10.1000/a'")
    conn.commit()
    conn.close()
    frozen.store = Store.open(frozen.db, AccessMode.READ_ONLY)
    frozen.write_config()

    with pytest.raises(PackageAuditError, match="10.1000/a: run 1 read other PDF bytes"):
        frozen.run()


def test_audit_refuses_run_without_pdf_hash(frozen):
    conn = frozen.reopen_writable()
    conn.execute("UPDATE extraction_run SET pdf_sha256 = NULL WHERE id = 1")
    frozen.refreeze(conn)
    with pytest.raises(PackageAuditError, match="run 1 has no PDF hash"):
        frozen.run()


def test_audit_refuses_guideline_without_recommendations(frozen):
    conn = frozen.reopen_writable()
    conn.execute("DELETE FROM recommendation WHERE run_id = 1")
    frozen.refreeze(conn)
    with pytest.raises(PackageAuditError, match="run 1 has no recommendations"):
        frozen.run()


def test_audit_refuses_member_not_in_manifest(frozen):
    rows = _read(frozen.world.manifest).splitlines(keepends=True)
    with open(frozen.world.manifest, "w") as f:
        f.writelines(r for r in rows if not r.startswith("10.1000/c,"))
    frozen.write_config(n_guidelines=3, n_current=3)
    with pytest.raises(PackageAuditError) as e:
        frozen.run()
    assert "10.1000/c: snapshot member not in the manifest" in e.value.problems


def test_audit_refuses_wal_sidecar(frozen):
    with open(frozen.db + "-wal", "wb") as f:
        f.write(b"frames the DB hash does not cover")
    with pytest.raises(PackageAuditError, match="-wal"):
        frozen.run()


def test_audit_refuses_uncommitted_code(frozen):
    with pytest.raises(PackageAuditError, match="uncommitted"):
        frozen.run(tree_state=TreeState.DIRTY)
    assert not os.path.exists(frozen.package)


@pytest.mark.parametrize("contents", [["evident.sqlite"], ["tables/"]])
def test_audit_refuses_a_target_that_is_not_a_package(frozen, contents):
    target = os.path.join(frozen.root, "elsewhere")
    for name in contents:
        path = os.path.join(target, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not name.endswith("/"):
            with open(path, "w") as f:
                f.write("keep me")
        else:
            os.makedirs(path, exist_ok=True)

    with pytest.raises(PackageAuditError, match="not a previous package"):
        frozen.run(package_dir=target)
    assert sorted(os.listdir(target)) == sorted(n.rstrip("/") for n in contents)


def test_audit_refuses_a_file_as_target(frozen):
    target = os.path.join(frozen.root, "file")
    with open(target, "w") as f:
        f.write("x")
    with pytest.raises(PackageAuditError, match="not a directory"):
        frozen.run(package_dir=target)


def test_empty_target_accepted(frozen):
    os.makedirs(frozen.package)
    frozen.run()
    assert _files(frozen.package) == _EXPECTED


def test_spotcheck_row_missing_from_run_refused(frozen):
    frozen._write_spotcheck(matched=(0, 99))
    with pytest.raises(ValueError, match="99"):
        frozen.run()


def test_editor_lock_files_ignored(frozen):
    for name in ("~$10.1000_e.xlsx", ".~lock.10.1000_e.xlsx#"):
        with open(os.path.join(frozen.heldout, name), "wb") as f:
            f.write(b"lock")
    frozen.run()
    manifest = json.loads(_read(os.path.join(frozen.package, "analysis_manifest.json")))
    assert list(manifest["inputs"]["heldout_labels"]) == ["10.1000_e.xlsx"]


def test_similarity_loaded_at_the_stored_embedding_revision(frozen):
    frozen.run()
    version = frozen.store.get_extractor_version(frozen.store.get_snapshot(1).extractor_version_id)
    assert frozen.revisions == [version.embedding_revision]


def test_audit_refuses_unpublished_snapshot(frozen):
    with pytest.raises(PackageAuditError, match="snapshot 9 is missing"):
        publish_package(frozen.deps(), 9, frozen.inputs())


def test_incomplete_heldout_labels_refused(frozen):
    write_heldout_xlsx(frozen.heldout, _HELDOUT, [])  # the blank template
    frozen.write_config()
    with pytest.raises(HeldoutLabelError, match="no labelled statement"):
        frozen.run()
    assert not os.path.exists(frozen.package)


def test_missing_validation_refused(frozen):
    frozen.store.close()
    conn = sqlite3.connect(frozen.db)
    conn.execute("DELETE FROM validation")
    conn.commit()
    conn.close()
    frozen.store = Store.open(frozen.db, AccessMode.READ_ONLY)
    frozen.write_config()
    with pytest.raises(ValidationMissingError):
        frozen.run()


def test_unknown_draft_field_keeps_previous_package(frozen, tmp_path):
    frozen.run()
    before = {n: _read(os.path.join(frozen.package, n), "rb") for n in _EXPECTED}

    bad = tmp_path / "bad.j2"
    bad.write_text("Strong: {{ table1_society.ALL.pct_strongest }}\n")
    with pytest.raises(DraftFieldError, match="pct_strongest"):
        frozen.run(draft_template=str(bad))

    assert _files(frozen.package) == _EXPECTED
    assert {n: _read(os.path.join(frozen.package, n), "rb") for n in _EXPECTED} == before
    assert not [n for n in os.listdir(frozen.root) if n.startswith(".package-")]


def test_render_failure_keeps_previous_outputs(frozen, monkeypatch):
    frozen.run()
    before = _read(os.path.join(frozen.package, "table1_society.csv"))

    def broken(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(publishing, "_write_package_captions", broken)
    with pytest.raises(RuntimeError):
        frozen.run()
    assert _read(os.path.join(frozen.package, "table1_society.csv")) == before
    assert _files(frozen.package) == _EXPECTED


def test_dashboard_failure_keeps_previous_package(frozen, monkeypatch):
    frozen.run()
    before = {n: _read(os.path.join(frozen.package, n), "rb") for n in _EXPECTED}
    _plant_duplicate(frozen)  # a changed source: a completed package would differ

    def broken(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(publishing, "_write_dashboard", broken)
    with pytest.raises(RuntimeError):
        frozen.run()
    assert {n: _read(os.path.join(frozen.package, n), "rb") for n in _EXPECTED} == before
    assert not [n for n in os.listdir(frozen.root) if n.startswith(".package-")]


def test_published_state_unchanged(frozen):
    frozen.run()
    assert frozen.store.get_snapshot(1).state == SnapshotState.PUBLISHED
    assert [v.id for v in [frozen.store.latest_validation(1)]] == [1]


def test_exploratory_themes_stay_out_of_the_package(frozen, tmp_path, monkeypatch):
    themes = tmp_path / "assignments.csv"
    themes.write_text("key,doi,society,method,phrase,theme\n")  # no rows: every rec unassigned, figure still drawn
    deps = frozen.deps
    monkeypatch.setattr(frozen, "deps", lambda: dataclasses.replace(deps(), themes_path=str(themes)))

    frozen.run()

    assert "Figure 5" in _read(os.path.join(frozen.world.publish_dir, "figures", "captions.md"))
    assert "Figure 5" not in _read(os.path.join(frozen.package, "captions.md"))
    assert "theme" not in _read(os.path.join(frozen.package, "draft.md")).lower()


# ── within-guideline duplicates ─────────────────────────────────

_SPOT_FIRST = GUIDELINES[_SPOT][3][0]  # (text, strength, certainty) of 10.1000/b row 0
_NOISY_COPY = "We suggest balanced crystal- loids rather than saline for resuscitation"
_PLANTED = (_SPOT, 0, 3)


def _plant_duplicate(frozen):
    """A hyphenation-noise copy of 10.1000/b row 0, as row 3 (rows 0–2 exist)."""
    conn = frozen.reopen_writable()
    (run_id,) = conn.execute("SELECT run_id FROM recommendation WHERE text = ?", (_SPOT_FIRST[0],)).fetchone()
    conn.execute("INSERT INTO recommendation (run_id, ordinal, text, raw_strength, raw_certainty, raw_category, "
                 "page, chunk_id) VALUES (?, 3, ?, ?, ?, 'graded', 2, 0)",
                 (run_id, _NOISY_COPY, _SPOT_FIRST[1], _SPOT_FIRST[2]))
    frozen.refreeze(conn)


def _write_verdicts(frozen, skip_planted=False, model="model", planted_model="model"):
    """A verdict per screened pair: the planted copy duplicate, every other pair distinct."""
    pairs = duplicate_candidates(frozen.store, 1, frozen.world.manifest, FakeEncoder())
    rows = [(p, Verdict.DUPLICATE, "reason", planted_model, "digest") if (p.doi, p.ordinal_a, p.ordinal_b) == _PLANTED
            else (p, Verdict.DISTINCT, "reason", model, "digest")
            for p in pairs if not (skip_planted and (p.doi, p.ordinal_a, p.ordinal_b) == _PLANTED)]
    path = os.path.join(frozen.root, "verdicts.csv")
    write_adjudications(path, rows)
    frozen.write_config(dedup_verdicts_path=path, dedup_verdicts_sha256=file_sha256(path))


def _esicm_graded(frozen):
    t1 = {r["society"]: r for r in _csv(os.path.join(frozen.package, "table1_society.csv"))}
    return t1["ESICM"]["n_graded"]


def test_planted_duplicate_counts_without_verdicts(frozen):
    _plant_duplicate(frozen)
    frozen.run()
    assert _esicm_graded(frozen) == "5"
    assert _csv(os.path.join(frozen.package, "duplicates.csv")) == []


def test_duplicate_removed_from_tables(frozen):
    _plant_duplicate(frozen)
    _write_verdicts(frozen)
    frozen.run()

    assert _esicm_graded(frozen) == "4"
    (removed,) = _csv(os.path.join(frozen.package, "duplicates.csv"))
    assert (removed["doi"], removed["kept_ordinal"], removed["removed_ordinal"]) == ("10.1000/b", "0", "3")
    facts = json.loads(_read(os.path.join(frozen.package, "methods_facts.json")))
    assert facts["dedup"]["n_removed"] == 1
    assert facts["dedup"]["removed_by_society"] == {"ESICM": 1}


def test_sensitivity_table_keeps_the_duplicates(frozen):
    _plant_duplicate(frozen)
    _write_verdicts(frozen)
    frozen.run()

    s2 = {r["society"]: r for r in _csv(os.path.join(frozen.package, "table_s2_without_dedup.csv"))}
    assert s2["ESICM"]["n_graded"] == "5"  # the removed copy counts again
    assert _esicm_graded(frozen) == "4"


def test_sensitivity_table_equals_table1_without_verdicts(frozen):
    frozen.run()
    assert (_read(os.path.join(frozen.package, "table_s2_without_dedup.csv"))
            == _read(os.path.join(frozen.package, "table1_society.csv")))


def test_methods_name_each_judge_and_count_split_pairs(frozen):
    _plant_duplicate(frozen)
    _write_verdicts(frozen, model="qwen+gemma", planted_model="qwen+gemma")
    frozen.run()

    facts = json.loads(_read(os.path.join(frozen.package, "methods_facts.json")))
    assert facts["dedup"]["adjudication_models"] == ["gemma", "qwen"]
    assert (facts["dedup"]["n_judges_split"], facts["dedup"]["n_blocked_by_grade"]) == (0, 0)


def test_spotcheck_row_removed_as_duplicate_maps_to_kept_copy(frozen):
    _plant_duplicate(frozen)
    _write_verdicts(frozen)
    frozen._write_spotcheck(matched=(3, 1))
    frozen.run()  # would raise "Spot-check rows missing" without the remap


def test_screened_pair_without_verdict_fails_closed(frozen):
    _plant_duplicate(frozen)
    _write_verdicts(frozen, skip_planted=True)
    with pytest.raises(DedupVerdictError, match="10.1000/b rows 0, 3: no verdict"):
        frozen.run()


def test_audit_refuses_changed_verdicts(frozen):
    _plant_duplicate(frozen)
    _write_verdicts(frozen)
    with open(os.path.join(frozen.root, "verdicts.csv"), "a") as f:
        f.write("\n")
    with pytest.raises(PackageAuditError, match="dedup verdicts"):
        frozen.run()


def test_outputs_show_the_deduplicated_development_validation(frozen):
    _plant_duplicate(frozen)
    conn = frozen.reopen_writable()
    (report_json,) = conn.execute("SELECT report_json FROM validation").fetchone()
    stale = json.loads(report_json)
    stale["candidate"]["f1"] = 0.123  # stored before duplicate removal
    conn.execute("UPDATE validation SET report_json = ?", (json.dumps(stale),))
    frozen.refreeze(conn)
    _write_verdicts(frozen)
    frozen.run()

    dev = json.loads(_read(os.path.join(frozen.package, "validation_development.json")))
    f1 = f"F1 {dev['report']['candidate']['f1']:.3f}"
    for path in (os.path.join(frozen.package, "captions.md"),
                 os.path.join(frozen.world.publish_dir, "figures", "captions.md"),
                 os.path.join(frozen.world.publish_dir, "dashboard", "index.html")):
        text = _read(path)
        assert "0.123" not in text, path
        assert f1 in text, path
