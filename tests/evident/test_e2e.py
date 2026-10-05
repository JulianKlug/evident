"""Exit test: the full command sequence on fakes (no GPU), spec §11.

    snapshot NEW (C fails) → RESUME → validate → publish → add D → snapshot NEW (worse)
        → validate FAIL → publish refused → publish --accept-regression → status clean
"""

import csv
import os

import pytest

from evident.domain import GateResult, MemberOrigin, SnapshotState
from evident.pipeline import (
    AddStatus,
    GateFailedError,
    SnapshotStart,
    add,
    publish,
    snapshot,
    status,
    validate,
)
from tests.evident.pipeline_fakes import CONFIG, WORSE_CONFIG, FakeRunner, World

_C, _D = "10.1000/c", "10.1000/d"


def test_full_sequence(tmp_path):
    world = World(tmp_path, runner=FakeRunner(fail=(_C,)))
    deps = world.deps()

    # 1. First build: C crashes
    first = snapshot(deps, SnapshotStart.NEW, CONFIG)
    assert first.state == SnapshotState.BUILDING
    s = status(deps)
    assert (s.n_failed, s.pending) == (1, [_C])
    assert s.eta_s is not None

    # 2. Resume: only C re-runs; the failure stays as history
    resumed = snapshot(deps, SnapshotStart.RESUME)
    assert resumed.state == SnapshotState.COMPLETE
    s = status(deps)
    assert (s.n_failed, s.pending, s.failures[0][0]) == (1, [], _C)

    # 3–4. Validate, publish v0
    assert validate(deps).gate.result == GateResult.NO_BASELINE
    v0 = publish(deps)
    assert not v0.republished
    assert world.store.get_snapshot(v0.snapshot_id).state == SnapshotState.PUBLISHED
    assert os.path.isfile(world.headline)
    assert os.path.isfile(os.path.join(world.publish_dir, "dashboard", "index.html"))

    # 5. A guideline published later joins v0, then outputs are rewritten
    world.add_row(_D)
    added = add(deps, os.path.join(world.pdf_dir, "10_1000_d.pdf"))
    assert added.status == AddStatus.ADDED
    gid_d = world.store.get_guideline(_D).id
    assert world.store.members(v0.snapshot_id)[-1].origin == MemberOrigin.POST_PUBLISH
    assert world.store.active_run(v0.snapshot_id, gid_d) is not None
    assert publish(deps).republished
    assert list(csv.DictReader(open(world.headline)))[-1]["n_guidelines"] == "4"
    assert _D in open(os.path.join(world.publish_dir, "dashboard", "index.html")).read()

    # 6–8. A worse extractor is refused by the gate
    worse = snapshot(deps, SnapshotStart.NEW, WORSE_CONFIG)
    assert worse.state == SnapshotState.COMPLETE
    assert validate(deps).gate.result == GateResult.FAIL
    with pytest.raises(GateFailedError):
        publish(deps)

    # 9. ...unless the regression is accepted with a reason
    accepted = publish(deps, accept_regression="test")
    assert world.store.get_snapshot(accepted.snapshot_id).accept_reason == "test"
    assert world.store.get_snapshot(accepted.snapshot_id).state == SnapshotState.PUBLISHED

    # 10. Clean
    s = status(deps)
    assert (s.snapshot_id, s.pending, s.coverage_gaps, s.n_interrupted) == (accepted.snapshot_id, [], [], 0)
    world.store.close()
