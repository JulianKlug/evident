"""Pipeline commands: snapshot, validate, publish, add, status.

    snapshot NEW ──▶ BUILDING ──(every included member has a run)──▶ COMPLETE
         ▲   │                                                         │
    RESUME   └── failed / interrupted runs stay pending                 ▼
                                               validate ──▶ gate ──▶ publish ──▶ PUBLISHED + headline.csv
                                                                                   │
                                                          add <pdf> (same version) ┘

Every model, Ollama call and GT load comes in through PipelineDeps, so tests run without a GPU.
"""

from __future__ import annotations

import logging
import os
import shutil
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence

from evident.analytics import HarmonizedRow, headline, write_headline
from evident.corpus import ManifestEntry, current_editions, load_manifest, sync_to_store
from evident.domain import (
    Category,
    ExtractorVersion,
    GateResult,
    Guideline,
    MemberOrigin,
    RawRecommendation,
    RunStatus,
    Snapshot,
    SnapshotState,
)
from evident.extraction import (
    Artifacts,
    ExtractionOutput,
    ExtractorConfig,
    PdfMissingError,
    check_artifacts,
    version_for,
)
from evident.harmonization import harmonize
from evident.store import Store
from evident.validation import GateReport, LabelledGuideline, SnapshotScore, gate, score

_log = logging.getLogger(__name__)

_SLOW_FACTOR = 3.0           # a run slower than 3 × median is flagged
_LOW_RECALL_FACTOR = 0.5     # fewer recs than half the society median is flagged
_UNGRADED_SHARE_MARGIN = 0.25  # ungraded share above society median + 0.25 is flagged
_DIGEST_PREFIX = 12  # same length as `ollama list` IDs


class SnapshotStart(str, Enum):
    NEW = "new"
    RESUME = "resume"


class IngestMode(str, Enum):
    NORMAL = "normal"  # a guideline already extracted is left alone
    FORCE = "force"    # extract again; the newer succeeded run becomes active


class AddStatus(str, Enum):
    ADDED = "added"
    ALREADY_PRESENT = "already_present"
    FAILED = "failed"


@dataclass(frozen=True)
class PipelineDeps:
    store: Store
    manifest_path: str
    pdf_dir: str
    runner: Callable[[ExtractorConfig, str, str], ExtractionOutput]  # (config, pdf path, doi)
    probe: Callable[[ExtractorConfig], Artifacts]
    labelled: Callable[[], list[LabelledGuideline]]
    similarity_model: object
    code_sha: Callable[[], Optional[str]]
    headline_path: str


# ── errors ──────────────────────────────────────────────────────

class PipelineError(Exception):
    """Base class for pipeline errors; the CLI prints the message, no traceback."""


class SnapshotInProgressError(PipelineError):
    def __init__(self, snapshot_id: int):
        super().__init__(f"Snapshot {snapshot_id} is still building; use --resume")
        self.snapshot_id = snapshot_id


class NoSnapshotInProgressError(PipelineError):
    def __init__(self):
        super().__init__("No building snapshot to resume")


class ConfigRequiredError(PipelineError):
    def __init__(self):
        super().__init__("A new snapshot needs a config")


class UnknownDoiError(PipelineError):
    def __init__(self, doi: str):
        super().__init__(f"{doi} is not a member of the snapshot")
        self.doi = doi


class NoSnapshotToValidateError(PipelineError):
    def __init__(self):
        super().__init__("No complete snapshot to validate")


class ValidationMissingError(PipelineError):
    def __init__(self, snapshot_id: int):
        super().__init__(f"Snapshot {snapshot_id} has no validation; run validate first")
        self.snapshot_id = snapshot_id


class CoverageError(PipelineError):
    def __init__(self, dois: list[str]):
        super().__init__(f"Manifest guideline(s) without an active run: {', '.join(dois)}")
        self.dois = dois


class GateFailedError(PipelineError):
    def __init__(self, snapshot_id: int, report_json: str):
        super().__init__(f"Snapshot {snapshot_id} regresses against the published version; "
                         "publish with --accept-regression REASON to override")
        self.snapshot_id = snapshot_id
        self.report_json = report_json


class AmbiguousPublishError(PipelineError):
    def __init__(self, snapshot_ids: list[int]):
        ids = ", ".join(str(i) for i in snapshot_ids)
        super().__init__(f"Several complete snapshots ({ids}); choose one with --snapshot ID "
                         "and retire the others with reject")
        self.snapshot_ids = snapshot_ids


class NotPublishableError(PipelineError):
    def __init__(self, snapshot_id: int, state: Optional[SnapshotState]):
        what = state.value if state else "missing"
        super().__init__(f"Snapshot {snapshot_id} is {what}; only a complete snapshot, "
                         "or the latest published one (re-publish), can be published")
        self.snapshot_id = snapshot_id
        self.state = state


class StaleValidationError(PipelineError):
    def __init__(self, snapshot_id: int, validated_against: Optional[int], published: Optional[int]):
        super().__init__(f"Snapshot {snapshot_id} was gated against snapshot {validated_against}, "
                         f"but {published} is published now; run validate --snapshot {snapshot_id}")
        self.snapshot_id = snapshot_id
        self.validated_against = validated_against
        self.published = published


class NothingToPublishError(PipelineError):
    def __init__(self):
        super().__init__("No complete or published snapshot")


class NoPublishedSnapshotError(PipelineError):
    def __init__(self):
        super().__init__("No published snapshot; add extracts with the published version")


class ManifestEntryMissingError(PipelineError):
    def __init__(self, filename: str):
        super().__init__(f"No manifest row with pdf_filename {filename}; add the manifest row first")
        self.filename = filename


# ── reports ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class SnapshotReport:
    snapshot_id: int
    state: SnapshotState
    n_succeeded: int
    n_failed: int
    n_failed_empty: int
    pending: list[str]
    failures: list[tuple[str, str]]  # (doi, error) of this invocation
    model: str = ""  # "<name>@<digest prefix>", e.g. "qwen3:14b@bdbd181c33f2"


@dataclass(frozen=True)
class ValidationReport:
    validation_id: int
    snapshot_id: int
    baseline_snapshot_id: Optional[int]
    gate: GateReport
    model: str = ""
    baseline_model: Optional[str] = None


@dataclass(frozen=True)
class PublishReport:
    snapshot_id: int
    republished: bool  # True when only the outputs of an already published snapshot were rewritten
    headline_path: str
    n_headline_guidelines: int
    n_skipped_excluded: int
    n_skipped_superseded: int
    model: str = ""


@dataclass(frozen=True)
class RejectReport:
    snapshot_id: int
    model: str
    reason: str


@dataclass(frozen=True)
class AddReport:
    snapshot_id: int
    doi: str
    status: AddStatus
    error: Optional[str] = None


@dataclass(frozen=True)
class StatusReport:
    snapshot_id: Optional[int] = None  # None: no snapshot yet
    state: Optional[SnapshotState] = None
    extractor_version_id: Optional[str] = None
    model: Optional[str] = None
    n_members: int = 0
    excluded: list[tuple[str, str]] = field(default_factory=list)
    n_succeeded: int = 0
    n_failed: int = 0
    n_failed_empty: int = 0
    n_interrupted: int = 0
    pending: list[str] = field(default_factory=list)
    eta_s: Optional[float] = None
    slow: list[tuple[str, float]] = field(default_factory=list)
    outliers: list[tuple[str, str]] = field(default_factory=list)
    coverage_gaps: list[str] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)  # every failed run, as history
    gate: Optional[GateResult] = None


# ── commands ────────────────────────────────────────────────────

def snapshot(deps: PipelineDeps, start: SnapshotStart, config: Optional[ExtractorConfig] = None,
             excludes: Sequence[tuple[str, str]] = ()) -> SnapshotReport:
    entries = _sync(deps)
    if start == SnapshotStart.NEW:
        snapshot_id, cfg = _new_snapshot(deps, config, entries)
    else:
        snapshot_id, cfg = _resume_snapshot(deps)

    for doi, reason in excludes:
        _exclude(deps.store, snapshot_id, doi, reason)

    tally = _Tally()
    code_sha = deps.code_sha()
    _extract_members(deps, snapshot_id, cfg, deps.store.pending_members(snapshot_id), code_sha, tally)

    # Re-scan: rows added to the manifest during the build join as RESCAN members
    while True:
        new_ids = _new_manifest_members(deps, snapshot_id)
        if not new_ids:
            break
        for gid in new_ids:
            deps.store.add_snapshot_member(snapshot_id, gid, MemberOrigin.RESCAN)
        _extract_members(deps, snapshot_id, cfg, new_ids, code_sha, tally)

    pending = deps.store.pending_members(snapshot_id)
    if not pending:
        deps.store.transition_snapshot(snapshot_id, SnapshotState.COMPLETE)

    return SnapshotReport(
        snapshot_id=snapshot_id,
        state=deps.store.get_snapshot(snapshot_id).state,
        n_succeeded=tally.succeeded, n_failed=len(tally.failures), n_failed_empty=tally.failed_empty,
        pending=_dois(deps.store, pending), failures=tally.failures,
        model=_model_label(deps.store, deps.store.get_snapshot(snapshot_id)),
    )


def validate(deps: PipelineDeps, snapshot_id: Optional[int] = None) -> ValidationReport:
    if snapshot_id is None:
        target = deps.store.latest_snapshot(SnapshotState.COMPLETE)
    else:
        target = deps.store.get_snapshot(snapshot_id)
    if target is None:
        raise NoSnapshotToValidateError()

    published = deps.store.latest_snapshot(SnapshotState.PUBLISHED)
    baseline = published if published and published.id != target.id else None

    labelled = deps.labelled()
    candidate_score = _score_snapshot(deps, target.id, labelled)
    baseline_score = _score_snapshot(deps, baseline.id, labelled) if baseline else None
    report = gate(candidate_score, baseline_score)

    baseline_id = baseline.id if baseline else None
    validation_id = deps.store.save_validation(target.id, baseline_id, report.result, report.to_json())
    return ValidationReport(validation_id, target.id, baseline_id, report, _model_label(deps.store, target),
                            _model_label(deps.store, baseline) if baseline else None)


def publish(deps: PipelineDeps, accept_regression: Optional[str] = None,
            snapshot_id: Optional[int] = None) -> PublishReport:
    """Publish a complete snapshot, or re-publish the latest published one's outputs.

    snapshot_id None: the only complete snapshot, else the published one; several → AmbiguousPublishError.
    """
    entries = load_manifest(deps.manifest_path)
    published = deps.store.latest_snapshot(SnapshotState.PUBLISHED)
    complete = _publish_target(deps.store, snapshot_id, published)

    # Re-publish: after `add`, only the outputs of the published snapshot change
    if complete.state == SnapshotState.PUBLISHED:
        return _write_outputs(deps, complete, entries, republished=True)

    validation = deps.store.latest_validation(complete.id)
    if validation is None:
        raise ValidationMissingError(complete.id)

    # The gate is only meaningful against the snapshot that is published now
    published_id = published.id if published else None
    if validation.baseline_snapshot_id != published_id:
        raise StaleValidationError(complete.id, validation.baseline_snapshot_id, published_id)

    gaps = _coverage_gaps(deps.store, complete.id, entries)
    if gaps:
        raise CoverageError(gaps)

    if validation.gate == GateResult.FAIL and not accept_regression:
        raise GateFailedError(complete.id, validation.report_json)

    deps.store.transition_snapshot(complete.id, SnapshotState.PUBLISHED, accept_regression)
    return _write_outputs(deps, deps.store.get_snapshot(complete.id), entries, republished=False)


def reject(deps: PipelineDeps, snapshot_id: int, reason: str) -> RejectReport:
    """Retire a building or complete candidate; it stays in the DB with its reason."""
    deps.store.reject_snapshot(snapshot_id, reason)
    snap = deps.store.get_snapshot(snapshot_id)
    return RejectReport(snapshot_id, _model_label(deps.store, snap), reason)


def add(deps: PipelineDeps, pdf_path: str, mode: IngestMode = IngestMode.NORMAL) -> AddReport:
    published = deps.store.latest_snapshot(SnapshotState.PUBLISHED)
    if published is None:
        raise NoPublishedSnapshotError()

    filename = os.path.basename(pdf_path)
    entry = next((e for e in load_manifest(deps.manifest_path) if e.pdf_filename == filename), None)
    if entry is None:
        raise ManifestEntryMissingError(filename)
    if not os.path.isfile(pdf_path):
        raise PdfMissingError(pdf_path)

    # The PDF dir is the single home of PDFs
    dest = os.path.join(deps.pdf_dir, filename)
    if os.path.abspath(pdf_path) != os.path.abspath(dest):
        os.makedirs(deps.pdf_dir, exist_ok=True)
        shutil.copyfile(pdf_path, dest)
    sync_to_store([entry], deps.store, deps.pdf_dir)  # DuplicatePdfError if the bytes belong to another DOI

    guideline = deps.store.get_guideline(entry.doi)
    if mode == IngestMode.NORMAL and deps.store.active_run(published.id, guideline.id):
        return AddReport(published.id, guideline.doi, AddStatus.ALREADY_PRESENT)

    # Same version as the published snapshot, or nothing: add never changes version
    cfg = _config_checked(deps, deps.store.get_extractor_version(published.extractor_version_id))
    if guideline.id not in {m.guideline_id for m in deps.store.members(published.id)}:
        deps.store.add_published_member(published.id, guideline.id)

    error = _extract(deps, published.id, cfg, guideline, deps.code_sha())
    if error:
        return AddReport(published.id, guideline.doi, AddStatus.FAILED, error)
    return AddReport(published.id, guideline.doi, AddStatus.ADDED)


def status(deps: PipelineDeps) -> StatusReport:
    store = deps.store
    latest = [store.latest_snapshot(s) for s in SnapshotState if s != SnapshotState.REJECTED]
    snap = max((s for s in latest if s), key=lambda s: s.id, default=None)
    if snap is None:
        return StatusReport()

    guidelines = _guidelines_by_id(store)
    members = store.members(snap.id)
    runs = store.runs(snap.id)
    pending = store.pending_members(snap.id)

    durations = [r.duration_s for r in runs if r.status == RunStatus.SUCCEEDED and r.duration_s is not None]
    median_s = statistics.median(durations) if durations else None
    validation = store.latest_validation(snap.id)

    return StatusReport(
        snapshot_id=snap.id,
        state=snap.state,
        extractor_version_id=snap.extractor_version_id,
        model=_model_label(store, snap),
        n_members=len(members),
        excluded=[(guidelines[m.guideline_id].doi, m.excluded_reason) for m in members if m.excluded_reason],
        n_succeeded=_count(runs, RunStatus.SUCCEEDED),
        n_failed=_count(runs, RunStatus.FAILED),
        n_failed_empty=_count(runs, RunStatus.FAILED_EMPTY),
        # Runs are sequential and finish or fail explicitly: a RUNNING row is a crash leftover
        n_interrupted=_count(runs, RunStatus.RUNNING),
        pending=[guidelines[gid].doi for gid in pending],
        eta_s=median_s * len(pending) if median_s is not None else None,
        slow=[(guidelines[r.guideline_id].doi, r.duration_s) for r in runs
              if r.status == RunStatus.SUCCEEDED and median_s and r.duration_s > _SLOW_FACTOR * median_s],
        outliers=_outliers(store, snap.id, guidelines),
        coverage_gaps=_coverage_gaps(store, snap.id, load_manifest(deps.manifest_path)),
        failures=[(guidelines[r.guideline_id].doi, r.error or "") for r in runs if r.status == RunStatus.FAILED],
        gate=validation.gate if validation else None,
    )


# ── snapshot internals ──────────────────────────────────────────

@dataclass
class _Tally:
    succeeded: int = 0
    failed_empty: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)


def _sync(deps: PipelineDeps) -> list[ManifestEntry]:
    entries = load_manifest(deps.manifest_path)
    sync_to_store(entries, deps.store, deps.pdf_dir)
    return entries


def _new_snapshot(deps: PipelineDeps, cfg: Optional[ExtractorConfig],
                  entries: list[ManifestEntry]) -> tuple[int, ExtractorConfig]:
    if cfg is None:
        raise ConfigRequiredError()
    building = deps.store.latest_snapshot(SnapshotState.BUILDING)
    if building:
        raise SnapshotInProgressError(building.id)

    version = version_for(cfg, deps.probe(cfg))
    deps.store.register_extractor_version(version)

    # All editions are members: trends need superseded ones too
    gids = [deps.store.get_guideline(e.doi).id for e in entries]
    snapshot_id = deps.store.create_snapshot(version.id, gids)
    _log.info("snapshot %d: new, version %s, model %s@%s, %d guidelines", snapshot_id, version.id[:12],
              version.model_name, version.model_digest[:_DIGEST_PREFIX], len(gids))
    return snapshot_id, cfg


def _resume_snapshot(deps: PipelineDeps) -> tuple[int, ExtractorConfig]:
    building = deps.store.latest_snapshot(SnapshotState.BUILDING)
    if building is None:
        raise NoSnapshotInProgressError()
    cfg = _config_checked(deps, deps.store.get_extractor_version(building.extractor_version_id))
    _log.info("snapshot %d: resume", building.id)
    return building.id, cfg


def _publish_target(store: Store, snapshot_id: Optional[int], published: Optional[Snapshot]) -> Snapshot:
    if snapshot_id is not None:
        snap = store.get_snapshot(snapshot_id)
        is_latest_published = snap is not None and published is not None and snap.id == published.id
        if snap is None or not (snap.state == SnapshotState.COMPLETE or is_latest_published):
            raise NotPublishableError(snapshot_id, snap.state if snap else None)
        return snap

    complete = store.complete_snapshots()
    if len(complete) > 1:
        raise AmbiguousPublishError([s.id for s in complete])
    if complete:
        return complete[0]
    if published is None:
        raise NothingToPublishError()
    return published


def _config_checked(deps: PipelineDeps, version: ExtractorVersion) -> ExtractorConfig:
    """Stored config, after refusing any artifact drift (model re-pull, classifier, BioLORD)."""
    cfg = ExtractorConfig.from_json(version.config_json)
    check_artifacts(version, deps.probe(cfg))
    return cfg


def _exclude(store: Store, snapshot_id: int, doi: str, reason: str) -> None:
    guideline = store.get_guideline(doi)
    if guideline is None or guideline.id not in {m.guideline_id for m in store.members(snapshot_id)}:
        raise UnknownDoiError(doi)
    store.exclude_member(snapshot_id, guideline.id, reason)


def _new_manifest_members(deps: PipelineDeps, snapshot_id: int) -> list[int]:
    entries = _sync(deps)
    members = {m.guideline_id for m in deps.store.members(snapshot_id)}
    gids = [deps.store.get_guideline(e.doi).id for e in entries]
    return [gid for gid in gids if gid not in members]


def _extract_members(deps: PipelineDeps, snapshot_id: int, cfg: ExtractorConfig, gids: list[int],
                     code_sha: Optional[str], tally: _Tally) -> None:
    guidelines = _guidelines_by_id(deps.store)
    for i, gid in enumerate(gids, start=1):
        g = guidelines[gid]
        _log.info("snapshot %d: [%d/%d] %s", snapshot_id, i, len(gids), g.doi)
        error = _extract(deps, snapshot_id, cfg, g, code_sha)
        if error:
            tally.failures.append((g.doi, error))
            continue
        if deps.store.active_run(snapshot_id, gid):
            tally.succeeded += 1
        else:
            tally.failed_empty += 1


def _extract(deps: PipelineDeps, snapshot_id: int, cfg: ExtractorConfig, g: Guideline,
             code_sha: Optional[str]) -> Optional[str]:
    """One run: start → runner → finish, or fail with the error. Returns the error, if any.

    A missing PDF fails the run; it is never auto-excluded, so the gap stays visible.
    """
    run_id = deps.store.start_run(snapshot_id, g.id, cfg.thinking, few_shot=[], code_sha=code_sha)
    try:
        pdf_path = g.pdf_path or ""
        if not os.path.isfile(pdf_path):
            raise PdfMissingError(pdf_path or f"(no PDF on disk for {g.doi})")
        out = deps.runner(cfg, pdf_path, g.doi)
    except Exception as e:  # KeyboardInterrupt/SystemExit propagate: the run stays RUNNING
        error = f"{type(e).__name__}: {e}"
        deps.store.fail_run(run_id, error)
        _log.warning("  failed: %s", error)
        return error

    run_status = deps.store.finish_run(run_id, out.recs, out.calls, out.n_pages, few_shot=out.few_shot)
    _log.info("  %s: %d recs, %d calls", run_status.value, len(out.recs), len(out.calls))
    return None


# ── read-side helpers ───────────────────────────────────────────

def _model_label(store: Store, snap: Snapshot) -> str:
    """Which LLM a snapshot ran on, e.g. "qwen3:14b@bdbd181c33f2" (name + digest prefix)."""
    version = store.get_extractor_version(snap.extractor_version_id)
    return f"{version.model_name}@{version.model_digest[:_DIGEST_PREFIX]}"


def _guidelines_by_id(store: Store) -> dict[int, Guideline]:
    return {g.id: g for g in store.list_guidelines()}


def _dois(store: Store, gids: list[int]) -> list[str]:
    guidelines = _guidelines_by_id(store)
    return [guidelines[gid].doi for gid in gids]


def _count(runs, status: RunStatus) -> int:
    return sum(r.status == status for r in runs)


def _active_recs(store: Store, snapshot_id: int) -> dict[int, list[RawRecommendation]]:
    """{guideline id: recs of the active run} for included members that have one."""
    out = {}
    for m in store.members(snapshot_id):
        if m.excluded_reason:
            continue
        run = store.active_run(snapshot_id, m.guideline_id)
        if run:
            out[m.guideline_id] = store.recommendations(run.id)
    return out


def _score_snapshot(deps: PipelineDeps, snapshot_id: int, labelled: list[LabelledGuideline]) -> SnapshotScore:
    guidelines = _guidelines_by_id(deps.store)
    recs_by_doi = {guidelines[gid].doi: recs for gid, recs in _active_recs(deps.store, snapshot_id).items()}
    return score(recs_by_doi, labelled, deps.similarity_model)


def _coverage_gaps(store: Store, snapshot_id: int, entries: list[ManifestEntry]) -> list[str]:
    """Manifest DOIs that are not members, or are included members without an active run."""
    members = {m.guideline_id: m for m in store.members(snapshot_id)}
    gaps = []
    for e in entries:
        g = store.get_guideline(e.doi)
        member = members.get(g.id) if g else None
        if member is None:
            gaps.append(e.doi)
            continue
        if not member.excluded_reason and store.active_run(snapshot_id, g.id) is None:
            gaps.append(e.doi)
    return gaps


def _outliers(store: Store, snapshot_id: int, guidelines: dict[int, Guideline]) -> list[tuple[str, str]]:
    """Low recall (few recs) or a high ungraded share, relative to the guideline's society."""
    stats = {}  # gid → (n recs, ungraded share)
    for gid, recs in _active_recs(store, snapshot_id).items():
        family = guidelines[gid].grading_family
        n_ungraded = sum(
            harmonize(r.raw_strength, r.raw_certainty, r.text, r.raw_category, family).category != Category.GRADED
            for r in recs)
        stats[gid] = (len(recs), n_ungraded / len(recs) if recs else 0.0)

    flagged = []
    for society in sorted({guidelines[gid].society for gid in stats}):
        peers = [gid for gid in stats if guidelines[gid].society == society]
        median_n = statistics.median(stats[gid][0] for gid in peers)
        median_share = statistics.median(stats[gid][1] for gid in peers)
        for gid in peers:
            n, share = stats[gid]
            if n < _LOW_RECALL_FACTOR * median_n:
                flagged.append((guidelines[gid].doi, f"low recall: {n} recs vs society median {median_n:g}"))
            if share > median_share + _UNGRADED_SHARE_MARGIN:
                flagged.append((guidelines[gid].doi, f"ungraded share {share:.2f} vs median {median_share:.2f}"))
    return flagged


def _write_outputs(deps: PipelineDeps, snap: Snapshot, entries: list[ManifestEntry],
                   republished: bool) -> PublishReport:
    """Headline over current editions only: included members with an active run."""
    guidelines = _guidelines_by_id(deps.store)
    current = current_editions(entries)
    n_excluded = sum(1 for m in deps.store.members(snap.id) if m.excluded_reason)

    rows = []
    contributing = set()
    n_superseded = 0
    for gid, recs in _active_recs(deps.store, snap.id).items():
        g = guidelines[gid]
        if g.doi not in current:
            n_superseded += 1
            continue
        contributing.add(g.doi)
        rows.extend(
            HarmonizedRow(g.doi, g.society, g.year,
                          harmonize(r.raw_strength, r.raw_certainty, r.text, r.raw_category, g.grading_family))
            for r in recs)

    model = _model_label(deps.store, snap)
    write_headline(headline(rows, snap.id, snap.extractor_version_id, model), deps.headline_path)
    return PublishReport(snap.id, republished, deps.headline_path, len(contributing), n_excluded, n_superseded,
                         model)
