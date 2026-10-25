"""Tests for evident.cli: parsing → pipeline calls with enums (pipeline patched), exit codes."""

import os
import sqlite3

import pytest

import evident.cli as cli
from evident import pipeline
from evident.domain import GradingFamily, Guideline, SnapshotState
from evident.pipeline import (
    AddReport,
    AddStatus,
    IngestMode,
    PackageReport,
    PublishReport,
    RejectReport,
    SnapshotReport,
    SnapshotStart,
)
from evident.corpus import file_sha256
from evident.store import Store
from evident.validation import heldout_report, SnapshotScore
from tests.evident.pipeline_fakes import CONFIG


@pytest.fixture
def calls(monkeypatch, tmp_path):
    """Patch every pipeline command; record (name, args)."""
    seen = []

    def record(name, result):
        def fake(deps, *args):
            seen.append((name, args))
            return result
        monkeypatch.setattr(pipeline, name, fake)

    record("snapshot", SnapshotReport(1, SnapshotState.COMPLETE, 3, 0, 0, [], []))
    record("publish", PublishReport(1, True, "out", 3, 0, 0))
    record("add", AddReport(1, "10.1000/d", AddStatus.ADDED))
    record("reject", RejectReport(3, "m@d", "worse F1"))
    return seen


def _main(tmp_path, *argv):
    return cli.main(["--db", str(tmp_path / "db.sqlite"), *argv])


def test_snapshot_new_with_excludes(calls, tmp_path):
    config = tmp_path / "v0.json"
    config.write_text(CONFIG.to_json())

    code = _main(tmp_path, "snapshot", "--config", str(config), "--exclude", "10.1/x", "--reason", "scan")

    assert code == 0
    assert calls == [("snapshot", (SnapshotStart.NEW, CONFIG, [("10.1/x", "scan")]))]


def test_snapshot_resume(calls, tmp_path):
    assert _main(tmp_path, "snapshot", "--resume") == 0
    assert calls == [("snapshot", (SnapshotStart.RESUME, None, []))]


@pytest.mark.parametrize("argv", [
    ["snapshot", "--resume", "--exclude", "10.1/x"],                # unpaired
    ["snapshot"],                                                   # neither --config nor --resume
    ["snapshot", "--resume", "--config", "c.json"],                 # both
])
def test_usage_errors_exit_2(calls, tmp_path, argv):
    with pytest.raises(SystemExit) as err:
        _main(tmp_path, *argv)
    assert err.value.code == 2
    assert calls == []


def test_add_force_then_publish(calls, tmp_path):
    assert _main(tmp_path, "add", "x.pdf", "--force") == 0
    # Re-publish the snapshot add attached to, never a pending candidate
    assert calls == [("add", ("x.pdf", IngestMode.FORCE)), ("publish", (None, 1))]


def test_failed_add_does_not_publish(calls, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "add", lambda deps, pdf, mode: AddReport(1, "d", AddStatus.FAILED, "boom"))
    assert _main(tmp_path, "add", "x.pdf") == 1
    assert calls == []


def test_publish_accept_regression(calls, tmp_path):
    assert _main(tmp_path, "publish", "--accept-regression", "noise") == 0
    assert calls == [("publish", ("noise", None))]


def test_publish_chosen_snapshot(calls, tmp_path):
    assert _main(tmp_path, "publish", "--snapshot", "2") == 0
    assert calls == [("publish", (None, 2))]


def test_reject(calls, tmp_path):
    assert _main(tmp_path, "reject", "--snapshot", "3", "--reason", "worse F1") == 0
    assert calls == [("reject", (3, "worse F1"))]


@pytest.mark.parametrize("argv", [["reject", "--snapshot", "3"], ["reject", "--reason", "x"]])
def test_reject_needs_id_and_reason(calls, tmp_path, argv):
    with pytest.raises(SystemExit) as err:
        _main(tmp_path, *argv)
    assert err.value.code == 2


def test_known_error_exit_1_message_only(calls, tmp_path, monkeypatch, capsys):
    def refuse(deps, reason, snapshot_id):
        raise pipeline.NothingToPublishError()
    monkeypatch.setattr(pipeline, "publish", refuse)

    assert _main(tmp_path, "publish") == 1
    err = capsys.readouterr().err
    assert "No complete or published snapshot" in err and "Traceback" not in err


def test_status_empty_db(tmp_path, capsys):
    assert _main(tmp_path, "status") == 0
    assert capsys.readouterr().out.strip() == "no snapshot"


def test_models_not_loaded_for_status(tmp_path, monkeypatch):
    def boom():
        raise AssertionError("BioLORD loaded")
    monkeypatch.setattr(cli, "load_similarity_model", boom)
    assert _main(tmp_path, "status") == 0



def test_repo_config_is_v0_baseline():
    cfg = cli._read_config(os.path.join(cli._REPO_ROOT, "configs", "v0.json"))
    assert (cfg.mode.value, cfg.model, cfg.thinking.value, cfg.prompt_version, cfg.few_shot.value) == (
        "single_pass", "qwen3:14b", "off", "v0", "few_shot")


# ── publish --package ───────────────────────────────────────────

_PAPER1 = os.path.join(cli._REPO_ROOT, "configs", "paper1.json")


@pytest.mark.parametrize("argv", [
    ["publish", "--snapshot", "8", "--package", "p"],                                         # no config/labels
    ["publish", "--snapshot", "8", "--package", "p", "--analysis-config", _PAPER1],          # no labels
    ["publish", "--package", "p", "--analysis-config", _PAPER1, "--heldout-labels", "h"],    # no snapshot
    ["publish", "--snapshot", "8", "--package", "p", "--analysis-config", _PAPER1, "--heldout-labels", "h",
     "--accept-regression", "x"],                                                           # no gate in a package
])
def test_package_usage_errors_exit_2(calls, tmp_path, argv):
    with pytest.raises(SystemExit) as err:
        _main(tmp_path, *argv)
    assert err.value.code == 2


def _frozen_db(tmp_path) -> str:
    path = str(tmp_path / "db.sqlite")
    Store.open(path).close()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode = DELETE")
    conn.close()
    return path


def test_package_reads_frozen_db_read_only(tmp_path, monkeypatch):
    seen = []
    loaded = []
    monkeypatch.setattr(cli, "load_similarity_model", lambda *args: loaded.append(args))
    monkeypatch.setattr(cli, "_tree_state", lambda: pipeline.TreeState.DIRTY)

    def fake(deps, snapshot_id, inputs):
        seen.append((snapshot_id, inputs))
        inputs.load_similarity("rev1")
        with pytest.raises(sqlite3.OperationalError):  # the store refuses writes
            deps.store.upsert_guideline(Guideline("10.1/a", "ACP", 2020, "T", "t", GradingFamily.GRADE))
        return PackageReport(8, "m@d", inputs.package_dir, [], "out", [], heldout_report(SnapshotScore(())))
    monkeypatch.setattr(pipeline, "publish_package", fake)
    db = _frozen_db(tmp_path)
    before = file_sha256(db)

    code = _main(tmp_path, "publish", "--snapshot", "8", "--package", "pkg", "--analysis-config", _PAPER1,
                 "--heldout-labels", "held")

    assert code == 0
    [(snapshot_id, inputs)] = seen
    assert (snapshot_id, inputs.db_path, inputs.package_dir, inputs.heldout_dir) == (8, db, "pkg", "held")
    assert inputs.config.name == "paper1"
    assert len(inputs.search_queries) == 9
    assert all(term.endswith('"2026/10/04"[dp])') for _, _, term in inputs.search_queries)
    assert inputs.tree_state == pipeline.TreeState.DIRTY  # the audit refuses it
    assert loaded == [("cpu", "rev1")]  # CPU, at the extraction runs' embedding revision
    assert file_sha256(db) == before


def test_package_missing_db_not_created(tmp_path, capsys):
    code = _main(tmp_path, "publish", "--snapshot", "8", "--package", "pkg", "--analysis-config", _PAPER1,
                 "--heldout-labels", "held")
    assert code == 1
    assert "No database" in capsys.readouterr().err
    assert not (tmp_path / "db.sqlite").exists()


def test_tree_state_from_git_status(monkeypatch):
    class Done:
        def __init__(self, stdout):
            self.stdout = stdout

    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: Done(" M evident/paper.py\n"))
    assert cli._tree_state() == pipeline.TreeState.DIRTY
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: Done(""))
    assert cli._tree_state() == pipeline.TreeState.CLEAN


def test_repo_paper1_config_matches_the_freeze():
    """docs/paper1/freeze.md: snapshot 8, 85 guidelines, 75 current, three prespecified SCCM DOIs."""
    with open(_PAPER1) as f:
        cfg = cli.AnalysisConfig.from_json(f.read())
    assert (cfg.snapshot_id, cfg.n_guidelines, cfg.n_current, cfg.search_cutoff) == (8, 85, 75, "2026-10-04")
    assert (cfg.bootstrap_n, cfg.bootstrap_seed) == (2000, 0)
    assert cfg.heldout_dois == ("10.1097/ccm.0000000000002533", "10.1097/ccm.0000000000006551",
                                "10.1097/pcc.0000000000003914")
    assert cfg.db_sha256 == "145ae06aab900753313fa9df79c91aa28bc4a32d836506b0ce78c2dd3b573bc0"
    assert cfg.heldout_labels_sha256 == {  # freeze.md, labels of 2026-10-19
        "10.1097_ccm.0000000000002533.xlsx": "113466fb6392049784829b81cd2f26a4136f00fe8f6a5b998bc9ae8011d0dc7f",
        "10.1097_ccm.0000000000006551.xlsx": "7fee54aa6c5fa5382779eceef928335fe4b10803c667c594fc70fdbba2fdd58f",
        "10.1097_pcc.0000000000003914.xlsx": "fdb835f7ca6ebaeb4711ab45171aaea7e6c2511639f3bfcb98082c5cdc2051eb",
    }
