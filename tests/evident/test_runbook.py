"""docs/runbook.md procedures A–C through the real CLI on a scratch DB (M5 spec §7 test 10).

    LiveWorld (fake runner, fake judges) ──cli.main──▶ add / publish / snapshot / validate / reject / status --check

The judge scripts call local LLMs; here LiveWorld.judge stands in for "both judges, then consensus".
"""

import dataclasses
import json
import os

import pytest

import evident.cli as cli
from evident.pipeline import SnapshotStart, publish, snapshot, validate
from evident.release import RELEASE_FILE
from tests.evident.pipeline_fakes import CONFIG, WORSE_CONFIG, LiveWorld

_C, _D, _F = "10.1000/c", "10.1000/d", "10.1000/f"
_F_COPY = (_F, 0, 2)
_OK, _ERROR, _NEEDS_REVIEW = 0, 1, 3


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Snapshot 1 published, as on the live machine after M4; the CLI gets the fake deps on its own connection."""
    w = LiveWorld(tmp_path)
    snapshot(w.live(), SnapshotStart.NEW, CONFIG)
    validate(w.live())
    w.judge(1)
    publish(w.live())
    monkeypatch.setattr(cli, "_deps", lambda args, store: dataclasses.replace(w.live(), store=store))
    yield w
    w.store.close()


def _cli(world, *argv) -> int:
    return cli.main(["--db", os.path.join(world.root, "db.sqlite"), "--manifest", world.manifest,
                     "--dedup-verdicts", world.ledger, *argv])


def _release(world) -> dict:
    with open(os.path.join(world.publish_dir, RELEASE_FILE)) as f:
        return json.load(f)


def test_a_add_a_guideline(world):
    first = _release(world)["release_id"]

    # Routine: one command when no new duplicate pair needs review
    world.add_guideline(_D)
    assert _cli(world, "add", world.pdf(_D)) == _OK
    assert _release(world)["previous_release_id"] == first

    # With a restated statement: ingested, outputs untouched, exit 3 until both judges ruled
    before = world.output_hashes()
    world.add_guideline(_F)
    assert _cli(world, "add", world.pdf(_F)) == _NEEDS_REVIEW
    assert world.output_hashes() == before
    assert _cli(world, "status", "--published", "--check") == _ERROR

    world.judge(1, duplicates=(_F_COPY,))
    assert _cli(world, "publish", "--snapshot", "1") == _OK
    assert _cli(world, "status", "--published", "--check") == _OK
    assert _release(world)["counts"]["current"] == 5

    # A second identical run is a no-op
    release, outputs = _release(world), world.output_hashes()
    assert _cli(world, "add", world.pdf(_F)) == _OK
    assert (_release(world), world.output_hashes()) == (release, outputs)


def test_b_add_a_new_edition(world):
    world.add_guideline(_F, supersedes=_C, topic="topic-c")
    assert _cli(world, "add", world.pdf(_F)) == _NEEDS_REVIEW
    world.judge(1, duplicates=(_F_COPY,))
    assert _cli(world, "publish", "--snapshot", "1") == _OK
    assert _cli(world, "status", "--published", "--check") == _OK

    counts = _release(world)["counts"]
    assert (counts["current"], counts["superseded"]) == (3, 1)
    with open(os.path.join(world.publish_dir, "tables", "trends.csv")) as f:
        trends = f.read()
    assert "2025" in trends and "2026" in trends  # the old edition stays in the history


def test_c_upgrade_the_extractor(world, tmp_path):
    def config(cfg, name):
        path = tmp_path / name
        path.write_text(cfg.to_json())
        return str(path)

    # A failed gate: the published release stays, the candidate is retired
    assert _cli(world, "snapshot", "--config", config(WORSE_CONFIG, "worse.json")) == _OK
    assert _cli(world, "validate", "--snapshot", "2") == _OK
    before = world.output_hashes()
    assert _cli(world, "publish", "--snapshot", "2") == _ERROR
    assert world.output_hashes() == before
    assert _cli(world, "reject", "--snapshot", "2", "--reason", "fewer recommendations") == _OK

    # A passing upgrade: same texts, so the ledger's verdicts still apply
    assert _cli(world, "snapshot", "--config", config(dataclasses.replace(CONFIG, seed=1), "seed1.json")) == _OK
    assert _cli(world, "validate", "--snapshot", "3") == _OK
    assert _cli(world, "publish", "--snapshot", "3") == _OK
    assert _cli(world, "status", "--published", "--check") == _OK
    assert _release(world)["snapshot"]["id"] == 3
