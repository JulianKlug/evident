"""Release record of a live publication (M5 spec §5): out/release.json, its id, the local archive.

    content (inputs, counts, output checksums) ─sha256─▶ release id
        │
        ├─ id == out/release.json id ──▶ UNCHANGED: outputs re-installed (same bytes), record untouched
        └─ new id ──▶ <archive>/<id>/release.json, <archive>/<id>/evident.db (SQLite backup)
                      ──▶ install outputs ──▶ out/release.json

The id hashes everything but the publication time and the previous id, so identical inputs and
outputs give the same id. The analysis-code SHA is part of the content: a new commit is a new release.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional


SCHEMA_VERSION = 1
RELEASE_FILE = "release.json"
ARCHIVE_DB = "evident.db"
_UNHASHED = ("release_id", "published_at", "previous_release_id")  # change on every publish, not the release


class ReleaseOutcome(str, Enum):
    NEW = "new"
    UNCHANGED = "unchanged"


def release_id(content: dict) -> str:
    """SHA-256 of the canonical JSON of the content, without the fields that change on every publish."""
    hashed = {k: v for k, v in content.items() if k not in _UNHASHED}
    return hashlib.sha256(_canonical(hashed).encode("utf-8")).hexdigest()


def current_release(publish_dir: str) -> Optional[dict]:
    """The published out/release.json, or None before the first live release."""
    path = os.path.join(publish_dir, RELEASE_FILE)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_release(content: dict, publish_dir: str, archive_dir: Optional[str], backup: Callable[[str], None],
                  install: Callable[[], None]) -> tuple[str, ReleaseOutcome]:
    """Archive, install() the outputs, then record the release; backup(path) writes the DB copy.

    The caller keeps the installed outputs only once this returns (StagedOutputs.keep/discard).

    The same release again re-installs the outputs (repairs a tampered out/) and keeps the record.
    """
    rid = release_id(content)
    previous = current_release(publish_dir)
    if previous is not None and previous.get("release_id") == rid:
        install()
        return rid, ReleaseOutcome.UNCHANGED

    record = {"schema_version": SCHEMA_VERSION, "release_id": rid,
              "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "previous_release_id": previous.get("release_id") if previous else None, **content}

    # Archive first: a failure leaves an archived release that out/ does not show yet, never the reverse
    if archive_dir is not None:
        release_dir = os.path.join(archive_dir, rid)
        os.makedirs(release_dir, exist_ok=True)
        _write_json(os.path.join(release_dir, RELEASE_FILE), record)
        backup(os.path.join(release_dir, ARCHIVE_DB))
    install()
    _write_json(os.path.join(publish_dir, RELEASE_FILE), record)
    return rid, ReleaseOutcome.NEW


def _write_json(path: str, data: dict) -> None:
    """Write <name>.part, then rename: a reader never sees half a record."""
    part = path + ".part"
    try:
        with open(part, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")
        os.replace(part, path)
    except BaseException:
        if os.path.exists(part):
            os.remove(part)
        raise


def _canonical(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
