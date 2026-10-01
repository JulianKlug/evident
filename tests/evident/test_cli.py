"""Tests for evident.cli: parsing → pipeline calls with enums (pipeline patched), exit codes."""

import pytest

import evident.cli as cli
from evident import pipeline
from evident.domain import SnapshotState
from evident.pipeline import AddReport, AddStatus, IngestMode, PublishReport, SnapshotReport, SnapshotStart
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
    record("publish", PublishReport(1, True, "out/headline.csv", 3, 0, 0))
    record("add", AddReport(1, "10.1000/d", AddStatus.ADDED))
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
    assert calls == [("add", ("x.pdf", IngestMode.FORCE)), ("publish", (None,))]


def test_failed_add_does_not_publish(calls, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "add", lambda deps, pdf, mode: AddReport(1, "d", AddStatus.FAILED, "boom"))
    assert _main(tmp_path, "add", "x.pdf") == 1
    assert calls == []


def test_publish_accept_regression(calls, tmp_path):
    assert _main(tmp_path, "publish", "--accept-regression", "noise") == 0
    assert calls == [("publish", ("noise",))]


def test_known_error_exit_1_message_only(calls, tmp_path, monkeypatch, capsys):
    def refuse(deps, reason):
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

