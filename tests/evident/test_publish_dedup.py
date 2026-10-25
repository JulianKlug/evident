"""Live publish applies the frozen duplicate verdicts, like publish --package; unjudged pairs keep both copies.

    World (b: 3 recs) ─snapshot─validate─▶ plant a copy of b row 0 as row 3 ─▶ publish(verdicts) ─▶ out/
"""

import csv
import dataclasses
import json
import os
import sqlite3

from evident.dedup import Verdict, write_adjudications
from evident.pipeline import SnapshotStart, duplicate_candidates, publish, snapshot, validate
from tests.evident.pipeline_fakes import CONFIG, GUIDELINES, World
from tests.legacy_golden import FakeEncoder

_DOIS = ("10.1000/a", "10.1000/b", "10.1000/c")
_COPY_OF = "10.1000/b"
_FIRST = GUIDELINES[_COPY_OF][3][0]  # (text, strength, certainty)
_COPY = "We suggest balanced crystal- loids rather than saline for resuscitation"
_PLANTED = (_COPY_OF, 0, 3)


def _world(tmp_path, stale_f1=None) -> World:
    """A validated, unpublished snapshot with a hyphenation-noise copy of 10.1000/b row 0."""
    world = World(tmp_path, dois=_DOIS)
    deps = world.deps()
    snapshot(deps, SnapshotStart.NEW, CONFIG, [])
    validate(deps)

    conn = sqlite3.connect(os.path.join(world.root, "db.sqlite"))
    (run_id,) = conn.execute("SELECT run_id FROM recommendation WHERE text = ?", (_FIRST[0],)).fetchone()
    conn.execute("INSERT INTO recommendation (run_id, ordinal, text, raw_strength, raw_certainty, raw_category, "
                 "page, chunk_id) VALUES (?, 3, ?, ?, ?, 'graded', 2, 0)", (run_id, _COPY, _FIRST[1], _FIRST[2]))
    if stale_f1 is not None:
        (report_json,) = conn.execute("SELECT report_json FROM validation").fetchone()
        report = json.loads(report_json)
        report["candidate"]["f1"] = stale_f1  # stored before duplicate removal
        conn.execute("UPDATE validation SET report_json = ?", (json.dumps(report),))
    conn.commit()
    conn.close()
    return world


def _verdicts(world: World, skip_planted=False) -> str:
    """The planted copy duplicate, every other screened pair distinct."""
    pairs = duplicate_candidates(world.store, 1, world.manifest, FakeEncoder())
    rows = [(p, Verdict.DUPLICATE if (p.doi, p.ordinal_a, p.ordinal_b) == _PLANTED else Verdict.DISTINCT,
             "reason", "m1+m2", "d1+d2")
            for p in pairs if not (skip_planted and (p.doi, p.ordinal_a, p.ordinal_b) == _PLANTED)]
    path = os.path.join(world.root, "verdicts.csv")
    write_adjudications(path, rows)
    return path


def _esicm_graded(world: World) -> str:
    with open(world.headline, newline="") as f:
        return {r["society"]: r for r in csv.DictReader(f)}["ESICM"]["n_graded"]


def test_publish_removes_judged_duplicates(tmp_path):
    world = _world(tmp_path)
    deps = dataclasses.replace(world.deps(), dedup_verdicts_path=_verdicts(world))
    report = publish(deps)

    assert _esicm_graded(world) == "4"  # b 3 + c 1; the copy is gone
    assert (report.n_duplicates_removed, report.n_unjudged_pairs) == (1, 0)


def test_publish_keeps_both_copies_of_an_unjudged_pair(tmp_path):
    world = _world(tmp_path)
    deps = dataclasses.replace(world.deps(), dedup_verdicts_path=_verdicts(world, skip_planted=True))
    report = publish(deps)

    assert _esicm_graded(world) == "5"
    assert (report.n_duplicates_removed, report.n_unjudged_pairs) == (0, 1)


def test_publish_without_verdicts_keeps_every_copy(tmp_path):
    world = _world(tmp_path)
    report = publish(world.deps())

    assert _esicm_graded(world) == "5"
    assert (report.n_duplicates_removed, report.n_unjudged_pairs) == (0, 0)


def test_dashboard_shows_the_deduplicated_validation(tmp_path):
    world = _world(tmp_path, stale_f1=0.123)
    deps = dataclasses.replace(world.deps(), dedup_verdicts_path=_verdicts(world))
    publish(deps)

    for path in (os.path.join(world.publish_dir, "figures", "captions.md"),
                 os.path.join(world.publish_dir, "dashboard", "index.html")):
        with open(path) as f:
            assert "0.123" not in f.read(), path
