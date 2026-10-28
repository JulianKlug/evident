"""Pipeline commands: snapshot, validate, publish, add, status.

    snapshot NEW ──▶ BUILDING ──(every included member has a run)──▶ COMPLETE
         ▲   │                                                         │
    RESUME   └── failed / interrupted runs stay pending                 ▼
                                               validate ──▶ gate ──▶ publish ──▶ PUBLISHED + tables, figures, dashboard
                                                                                   │
                                                          add <pdf> (same version) ┘

    publish (live, M5): gates ─▶ duplicate ledger ─▶ staged render ─▶ [transition] ─▶ archive ─▶ swap ─▶ out/release.json
        gates: candidates = manifest, every guideline has an active run, PDFs as the runs read them,
               extractor artefacts unchanged, search window; any failure → nothing written
    rebind-pdf <pdf>: new bytes, same prompts (no model call) ─▶ the run now names the new bytes

    publish_package (read-only, frozen copy of a PUBLISHED snapshot):
        audit (fail closed) ─▶ rows + held-out score + spot check ─▶ paper tables ─▶ <package>/ + outputs

Every model, Ollama call and GT load comes in through PipelineDeps, so tests run without a GPU.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import statistics
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Callable, Optional, Sequence

from evaluation.matching import build_similarity_matrix
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
from evident.corpus import (
    Candidate,
    ManifestEntry,
    ManifestError,
    check_candidates,
    current_editions,
    file_sha256,
    load_candidates,
    load_manifest,
    load_search_window,
    load_theme_assignments,
    sync_to_store,
)
from evident.dedup import (
    JUDGE_PROMPT,
    MODEL_SEPARATOR,
    SCREEN_BIOLORD,
    SPLIT_REASON,
    SCREEN_LEXICAL,
    Adjudication,
    CandidatePair,
    DedupResult,
    DedupVerdictError,
    Verdict,
    check_complete,
    deduplicate,
    load_adjudications,
    load_human_check,
    remap_ordinals,
    removed_rows,
    resolve,
    score_human_check,
    screen,
)
from evident.domain import (
    Category,
    ExtractionRun,
    ExtractorVersion,
    GateResult,
    Guideline,
    LlmCall,
    MemberOrigin,
    RawRecommendation,
    RunStatus,
    Snapshot,
    SnapshotState,
)
from evident.extraction import (
    ArtifactMismatchError,
    Artifacts,
    ExtractionOutput,
    ExtractorConfig,
    PdfMissingError,
    check_artifacts,
    prompt_template,
    version_for,
)
from evident.harmonization import harmonize
from evident.paper import (
    EDITION_RULE,
    AnalysisConfig,
    PaperPackage,
    corpus_flow,
    excluded_guidelines,
    figure_data,
    guideline_mean,
    methods_facts,
    raw_mapping,
    spotcheck_bias,
    table1,
    table2,
    trend_points,
)
from evident.publishing import (PublishBundle, StagedOutputs, ValidationSummary, package_target_problem,
                                render_package, stage)
from evident.release import ReleaseOutcome, current_release, write_release
from evident.store import DuplicatePdfError, Store
from evident.themes import UNASSIGNED, rec_key, theme_certainty
from evident.validation import (
    GateReport,
    HeldoutReport,
    LabelledGuideline,
    SnapshotScore,
    gate,
    heldout_report,
    load_heldout,
    load_spotcheck,
    score,
)
from utils.doi import normalize_doi

_log = logging.getLogger(__name__)

_SLOW_FACTOR = 3.0           # a run slower than 3 × median is flagged
_LOW_RECALL_FACTOR = 0.5     # fewer recs than half the society median is flagged
_UNGRADED_SHARE_MARGIN = 0.25  # ungraded share above society median + 0.25 is flagged
_DIGEST_PREFIX = 12  # same length as `ollama list` IDs
_SHA_SUFFIX = "_sha256"  # release input keys, e.g. manifest_sha256 → "manifest"
_BOOTSTRAP_UNIT = "guideline, resampled within society; ALL stratified by society"
_WAL_SUFFIX = "-wal"  # SQLite write-ahead log beside the DB: readers see its frames, the DB hash does not
_IGNORED_FILE_PREFIXES = ("~$", ".")  # editor lock files beside label files, e.g. .~lock.x.xlsx#
_REBIND_LOG_COLUMNS = ("doi", "run_id", "old_pdf_sha256", "new_pdf_sha256", "rebound_at")


class TreeState(str, Enum):
    """Git working tree of the analysis code; a package needs CLEAN so analysis_sha names the code."""
    CLEAN = "clean"
    DIRTY = "dirty"


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


class RebindStatus(str, Enum):
    REBOUND = "rebound"      # new bytes, same prompts: the run now names them
    RESTORED = "restored"    # the bytes the run read, copied back over a drifted PDF dir copy
    UNCHANGED = "unchanged"  # the bytes the run read, already in place


class StatusTarget(str, Enum):
    LATEST = "latest"        # newest snapshot not rejected, e.g. a candidate still building
    PUBLISHED = "published"  # the public release, even while a candidate builds


class StatusCheck(str, Enum):
    OFF = "off"
    ON = "on"  # run every publication gate; screens duplicates, so it loads BioLORD


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
    publish_dir: str  # tables/, figures/, dashboard/ are (re)written here
    themes_path: Optional[str] = None  # frozen theme assignments; None → no theme figure
    dedup_verdicts_path: Optional[str] = None  # live verdict ledger; None → every copy kept
    candidates_path: Optional[str] = None  # included candidates must equal the manifest; None → not checked
    search_path: Optional[str] = None  # corpus/search.json, the search window; None → not checked or recorded
    release_archive_dir: Optional[str] = None  # one dir per new release (record + DB backup); None → no archive
    prompt_hashes: Optional[Callable[[ExtractorConfig, str, str], dict[int, str]]] = None  # rebind-pdf; no LLM
    rebind_log_path: Optional[str] = None  # CSV appended per rebind: doi, run, old and new PDF SHA-256


@dataclass(frozen=True)
class PackageInputs:
    """What `publish --package` reads besides the store and manifest; all read-only."""
    db_path: str
    package_dir: str
    config: AnalysisConfig
    config_path: str
    heldout_dir: str
    search_queries: tuple[tuple[str, str, str], ...]  # (society, source, term with date window)
    tree_state: TreeState
    load_similarity: Callable[[str], object]  # embedding revision → BioLORD for held-out matching
    draft_template: Optional[str] = None  # manuscript template; None → package without draft.md


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


class PackageAuditError(PipelineError):
    """The analysis source is not the one the package was specified on; nothing is written."""

    def __init__(self, problems: list[str]):
        super().__init__("Analysis source failed the audit:\n" + "\n".join(f"  {p}" for p in problems))
        self.problems = problems


class PublishBlockedError(PipelineError):
    """A publication gate failed; nothing was written, the previous outputs and release stand."""

    def __init__(self, problems: list[str]):
        super().__init__("Publication blocked; previous outputs unchanged:\n" + "\n".join(f"  {p}" for p in problems))
        self.problems = problems


class UnresolvedDuplicatesError(PipelineError):
    """Screened pairs without a verdict for their texts; judge them, then publish again."""

    def __init__(self, snapshot_id: int, pairs: list[CandidatePair]):
        super().__init__(f"{len(pairs)} screened duplicate pair(s) of snapshot {snapshot_id} have no verdict for "
                         "their texts; previous outputs unchanged")
        self.snapshot_id = snapshot_id
        self.pairs = pairs


class PdfChangedError(PipelineError):
    def __init__(self, doi: str):
        super().__init__(f"PDF of {doi} differs from the bytes its active run read; restore them, or run "
                         "rebind-pdf (same model input) or add --force (re-extract)")
        self.doi = doi


class NoActiveRunError(PipelineError):
    def __init__(self, doi: str, snapshot_id: int):
        super().__init__(f"{doi} has no active run in published snapshot {snapshot_id}; use add")
        self.doi = doi


class PromptMismatchError(PipelineError):
    def __init__(self, doi: str, detail: str):
        super().__init__(f"New PDF of {doi} changes the model input ({detail}); nothing changed. "
                         "Re-extract with add --force")
        self.doi = doi
        self.detail = detail


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
    publish_dir: str
    n_headline_guidelines: int
    n_skipped_excluded: int
    n_skipped_superseded: int
    model: str = ""
    paths: list[str] = field(default_factory=list)  # every file written
    n_duplicates_removed: int = 0
    release_id: Optional[str] = None
    release_outcome: Optional[ReleaseOutcome] = None  # UNCHANGED: identical to the release already out


@dataclass(frozen=True)
class PackageReport:
    snapshot_id: int
    model: str
    package_dir: str
    package_paths: list[str]
    publish_dir: str
    paths: list[str]
    heldout: HeldoutReport


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
class RebindReport:
    snapshot_id: int
    doi: str
    status: RebindStatus
    old_sha256: Optional[str]
    new_sha256: str


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
    pdf_changed: list[str] = field(default_factory=list)  # PDF bytes differ from what the active run read
    n_current: int = 0       # manifest editions counted in the current map
    n_superseded: int = 0    # trends only
    release_id: Optional[str] = None  # out/release.json
    blockers: list[str] = field(default_factory=list)  # StatusCheck.ON: why publish would refuse; [] = clean
    n_unresolved_pairs: Optional[int] = None  # StatusCheck.ON only
    n_unassigned: Optional[int] = None  # current recs under "Not yet assigned" in the theme figure


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
    Every gate runs before anything is written (M5 spec §5): a failure leaves the outputs and release as they were.
    """
    entries = _sync(deps)  # current PDF hashes, for the changed-PDF gate
    published = deps.store.latest_snapshot(SnapshotState.PUBLISHED)
    target = _publish_target(deps.store, snapshot_id, published)

    # Re-publish: after `add`, only the outputs of the published snapshot change
    republished = target.state == SnapshotState.PUBLISHED
    validation = deps.store.latest_validation(target.id)
    if not republished:
        if validation is None:
            raise ValidationMissingError(target.id)

        # The gate is only meaningful against the snapshot that is published now
        published_id = published.id if published else None
        if validation.baseline_snapshot_id != published_id:
            raise StaleValidationError(target.id, validation.baseline_snapshot_id, published_id)

    # A re-publish passes the same coverage check: `add` can leave a manifest row without a run
    gaps = _coverage_gaps(deps.store, target.id, entries)
    if gaps:
        raise CoverageError(gaps)
    problems = _gate_problems(deps, target, entries)
    if problems:
        raise PublishBlockedError(problems)

    if not republished and validation.gate == GateResult.FAIL and not accept_regression:
        raise GateFailedError(target.id, validation.report_json)

    # Stage, transition, archive, swap in, name the release, then drop the previous outputs.
    # Any failure leaves the previous outputs and record; after the transition status blocks until published again.
    outputs = _live_outputs(deps, target, entries)
    staged = stage(outputs.bundle, deps.publish_dir)
    try:
        if not republished:
            deps.store.transition_snapshot(target.id, SnapshotState.PUBLISHED, accept_regression)
        report = _released(deps, deps.store.get_snapshot(target.id), entries, outputs, staged, republished)
        staged.keep()
        return report
    finally:
        staged.discard()


def reject(deps: PipelineDeps, snapshot_id: int, reason: str) -> RejectReport:
    """Retire a building or complete candidate; it stays in the DB with its reason."""
    deps.store.reject_snapshot(snapshot_id, reason)
    snap = deps.store.get_snapshot(snapshot_id)
    return RejectReport(snapshot_id, _model_label(deps.store, snap), reason)


def publish_package(deps: PipelineDeps, snapshot_id: int, inputs: PackageInputs) -> PackageReport:
    """Render a paper's analysis package from a frozen copy of a published snapshot.

    Never writes the store (no sync, no transition, no validation row): open it READ_ONLY.
    Every check fails closed before anything is rendered; the package dir and the standard outputs
    are both staged before either is swapped in.
    """
    cfg = inputs.config
    entries = load_manifest(deps.manifest_path)
    candidates = load_candidates(cfg.candidates_path)
    snap = deps.store.get_snapshot(snapshot_id)
    _audit_package(deps.store, snapshot_id, deps.manifest_path, entries, candidates, inputs)

    development = deps.store.latest_validation(snap.id)
    if development is None:
        raise ValidationMissingError(snap.id)

    # Held-out labels: exactly the prespecified DOIs, never the development set (HeldoutLabelError)
    development_dois = {lg.doi for lg in deps.labelled()}
    heldout = load_heldout(inputs.heldout_dir, cfg.heldout_dois, development_dois)

    all_rows, facts = _harmonized_rows(deps.store, snap, entries)
    version = deps.store.get_extractor_version(snap.extractor_version_id)
    similarity = inputs.load_similarity(version.embedding_revision)  # same embeddings as the runs

    # Within-guideline copies leave every table and both validation scores (spec §5)
    dedup, dedup_facts = _deduplicate(all_rows, similarity, cfg)
    rows = dedup.kept
    recs_by_doi = _kept_recs_by_doi(deps.store, snap.id, dedup.removed)
    heldout_scores = heldout_report(score({lg.doi: recs_by_doi[lg.doi] for lg in heldout}, heldout, similarity))
    development_report = json.loads(development.report_json)
    if dedup_facts is not None:
        development_report = _rescored_development(deps, recs_by_doi, similarity)
    spot = [replace(g, matched_ordinals=remap_ordinals(g.doi, g.matched_ordinals, dedup.removed))
            for g in load_spotcheck(cfg.spotcheck_dir)]

    current = current_editions(entries)
    t1, t2 = table1(rows, cfg.bootstrap_n, cfg.bootstrap_seed), table2(rows)
    society_of = {e.doi: e.society for e in entries}
    paper = PaperPackage(
        config=cfg,
        table1=t1,
        table2=t2,
        guideline_mean=guideline_mean(rows),
        figure_data=figure_data(t1, t2, trend_points(rows)),
        corpus_flow=corpus_flow(candidates, entries, current),
        excluded=excluded_guidelines(candidates),
        raw_mapping=raw_mapping(rows),
        spotcheck_bias=spotcheck_bias(rows, spot, cfg.bootstrap_n, cfg.bootstrap_seed),
        validation_development=development_report,
        validation_heldout=heldout_scores,
        heldout_societies=tuple(sorted({society_of[d] for d in (lg.doi for lg in heldout)})),
        methods_facts=methods_facts(cfg, entries, current, candidates, rows, inputs.search_queries, version,
                                    len(development_dois), dedup_facts),
        provenance=_provenance(deps, snap, version, inputs),
        duplicates=removed_rows(all_rows, dedup),
        draft_template=inputs.draft_template,
        table1_without_dedup=table1(all_rows, cfg.bootstrap_n, cfg.bootstrap_seed),  # bounds verdict errors
        **_model_specification(deps.store, snap, version),
    )

    # Captions and dashboard show the development scores the package reports, after duplicate removal
    validation = _summary_of(development_report, development.gate, development.baseline_snapshot_id)
    bundle = replace(_bundle(deps, snap, rows, facts, paper), validation=validation)
    package_paths, paths = render_package(bundle, inputs.package_dir, deps.publish_dir)
    return PackageReport(snap.id, bundle.model, inputs.package_dir, package_paths, deps.publish_dir, paths,
                         heldout_scores)


def duplicate_candidates(store: Store, snapshot_id: int, manifest_path: str, similarity) -> list[CandidatePair]:
    """Screened within-guideline pairs of a snapshot (all editions), for offline adjudication; read-only."""
    snap = store.get_snapshot(snapshot_id)
    rows, _ = _harmonized_rows(store, snap, load_manifest(manifest_path))
    return _screen_rows(rows, similarity)


def add(deps: PipelineDeps, pdf_path: str, mode: IngestMode = IngestMode.NORMAL) -> AddReport:
    published = deps.store.latest_snapshot(SnapshotState.PUBLISHED)
    if published is None:
        raise NoPublishedSnapshotError()

    entries = load_manifest(deps.manifest_path)
    filename = os.path.basename(pdf_path)
    entry = next((e for e in entries if e.pdf_filename == filename), None)
    if entry is None:
        raise ManifestEntryMissingError(filename)
    if not os.path.isfile(pdf_path):
        raise PdfMissingError(pdf_path)

    # Curation first: an extraction for a guideline publish would refuse is wasted GPU time
    problems = _candidate_problems(deps, entries) + _edition_problems(entries)
    if problems:
        raise PublishBlockedError(problems)

    # A changed PDF is never "already present", and must not overwrite the bytes the run read
    existing = deps.store.get_guideline(entry.doi)
    run = deps.store.active_run(published.id, existing.id) if existing else None
    new_sha = file_sha256(pdf_path)
    if mode == IngestMode.NORMAL and run and run.pdf_sha256 and run.pdf_sha256 != new_sha:
        raise PdfChangedError(entry.doi)
    owner = next((g.doi for g in deps.store.list_guidelines() if g.pdf_sha256 == new_sha and g.doi != entry.doi), None)
    if owner:
        raise DuplicatePdfError(owner, entry.doi)

    # Same version as the published snapshot, or nothing: add never changes version
    cfg = _config_checked(deps, deps.store.get_extractor_version(published.extractor_version_id))

    # The PDF dir is the single home of PDFs
    dest = os.path.join(deps.pdf_dir, filename)
    if os.path.abspath(pdf_path) != os.path.abspath(dest):
        os.makedirs(deps.pdf_dir, exist_ok=True)
        shutil.copyfile(pdf_path, dest)
    sync_to_store([entry], deps.store, deps.pdf_dir)  # DuplicatePdfError if the bytes belong to another DOI

    guideline = deps.store.get_guideline(entry.doi)
    if mode == IngestMode.NORMAL and deps.store.active_run(published.id, guideline.id):
        return AddReport(published.id, guideline.doi, AddStatus.ALREADY_PRESENT)

    if guideline.id not in {m.guideline_id for m in deps.store.members(published.id)}:
        deps.store.add_published_member(published.id, guideline.id)

    error = _extract(deps, published.id, cfg, guideline, deps.code_sha())
    if error:
        return AddReport(published.id, guideline.doi, AddStatus.FAILED, error)
    return AddReport(published.id, guideline.doi, AddStatus.ADDED)


def rebind_pdf(deps: PipelineDeps, pdf_path: str) -> RebindReport:
    """Accept new bytes for a guideline's PDF when they give the model the same prompts; no model call.

    Every prompt of the active run is rebuilt from the new PDF and compared by chunk with the stored hashes.
    Any difference (text, chunk count) changes nothing: PromptMismatchError, re-extract with add --force.
    """
    published = deps.store.latest_snapshot(SnapshotState.PUBLISHED)
    if published is None:
        raise NoPublishedSnapshotError()

    filename = os.path.basename(pdf_path)
    entry = next((e for e in load_manifest(deps.manifest_path) if e.pdf_filename == filename), None)
    if entry is None:
        raise ManifestEntryMissingError(filename)
    if not os.path.isfile(pdf_path):
        raise PdfMissingError(pdf_path)

    guideline = deps.store.get_guideline(entry.doi)
    run = deps.store.active_run(published.id, guideline.id) if guideline else None
    if run is None:
        raise NoActiveRunError(entry.doi, published.id)

    new_sha = file_sha256(pdf_path)
    dest = os.path.join(deps.pdf_dir, filename)
    if new_sha == run.pdf_sha256:
        if _sha_or_none(dest) == new_sha:
            return RebindReport(published.id, entry.doi, RebindStatus.UNCHANGED, run.pdf_sha256, new_sha)
        shutil.copyfile(pdf_path, dest)  # e.g. from a backup: restores what the run read
        return RebindReport(published.id, entry.doi, RebindStatus.RESTORED, run.pdf_sha256, new_sha)

    # Same version, same artefacts: the prompts rebuilt now are the ones the run would send
    cfg = _config_checked(deps, deps.store.get_extractor_version(published.extractor_version_id))
    mismatch = _prompt_mismatch(deps.prompt_hashes(cfg, pdf_path, entry.doi), deps.store.llm_calls(run.id))
    if mismatch:
        raise PromptMismatchError(entry.doi, mismatch)

    # Verified: the PDF dir gets the new bytes, guideline and run name them
    deps.store.rebind_pdf(guideline.id, run.id, dest, new_sha)  # DuplicatePdfError before anything is copied
    if os.path.abspath(pdf_path) != os.path.abspath(dest):
        shutil.copyfile(pdf_path, dest)
    _log_rebind(deps.rebind_log_path, entry.doi, run.id, run.pdf_sha256, new_sha)
    return RebindReport(published.id, entry.doi, RebindStatus.REBOUND, run.pdf_sha256, new_sha)


def status(deps: PipelineDeps, target: StatusTarget = StatusTarget.LATEST,
           check: StatusCheck = StatusCheck.OFF) -> StatusReport:
    store = deps.store
    if target == StatusTarget.PUBLISHED:
        snap = store.latest_snapshot(SnapshotState.PUBLISHED)
    else:
        latest = [store.latest_snapshot(s) for s in SnapshotState if s != SnapshotState.REJECTED]
        snap = max((s for s in latest if s), key=lambda s: s.id, default=None)
    if snap is None:
        return StatusReport(blockers=["no published snapshot"] if check == StatusCheck.ON else [])

    entries = _sync(deps)  # current PDF hashes, for the changed-PDF flag
    n_current = len(current_editions(entries))
    release = current_release(deps.publish_dir)
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
        pdf_changed=_pdf_changed(store, snap.id, guidelines),
        n_current=n_current,
        n_superseded=len(entries) - n_current,
        release_id=release.get("release_id") if release else None,
        **(_checked(deps, snap, entries, release) if check == StatusCheck.ON else {}),
    )


# ── snapshot internals ──────────────────────────────────────────

@dataclass
class _Tally:
    succeeded: int = 0
    failed_empty: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)


def _sync(deps: PipelineDeps) -> list[ManifestEntry]:
    entries = load_manifest(deps.manifest_path)
    report = sync_to_store(entries, deps.store, deps.pdf_dir)
    if report.changed_pdf:
        _log.warning("PDF bytes changed since the last sync: %s", ", ".join(report.changed_pdf))
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
    run_id = deps.store.start_run(snapshot_id, g.id, cfg.thinking, few_shot=[], code_sha=code_sha,
                                  pdf_sha256=g.pdf_sha256)
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


def _pdf_changed(store: Store, snapshot_id: int, guidelines: dict[int, Guideline]) -> list[str]:
    """DOIs whose active run read different PDF bytes than the file now synced (runs without a hash skipped)."""
    changed = []
    for m in store.members(snapshot_id):
        run = store.active_run(snapshot_id, m.guideline_id)
        g = guidelines[m.guideline_id]
        if run and run.pdf_sha256 and run.pdf_sha256 != g.pdf_sha256:
            changed.append(g.doi)
    return changed


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


@dataclass(frozen=True)
class _LiveOutputs:
    """What one live publish renders, before anything is written."""
    bundle: PublishBundle
    all_rows: list[HarmonizedRow]  # every edition, before duplicate removal
    dedup: DedupResult


def _live_outputs(deps: PipelineDeps, snap: Snapshot, entries: list[ManifestEntry]) -> _LiveOutputs:
    """Rows, duplicate removal and bundle; UnresolvedDuplicatesError when a screened pair has no verdict."""
    all_rows, facts = _harmonized_rows(deps.store, snap, entries)

    # Same duplicate removal as publish --package; a pair without a verdict for its texts blocks
    adjudications, unresolved = _live_verdicts(all_rows, deps)
    if unresolved:
        raise UnresolvedDuplicatesError(snap.id, unresolved)
    dedup = deduplicate(all_rows, adjudications)

    bundle = replace(_bundle(deps, snap, dedup.kept, facts), search_cutoff=_search_cutoff(deps))
    if dedup.removed:
        stored = deps.store.latest_validation(snap.id)
        report = _rescored_development(deps, _kept_recs_by_doi(deps.store, snap.id, dedup.removed),
                                       deps.similarity_model)
        bundle = replace(bundle, validation=_summary_of(report, stored.gate, stored.baseline_snapshot_id)
                         if stored else None)
    return _LiveOutputs(bundle, all_rows, dedup)


def _released(deps: PipelineDeps, snap: Snapshot, entries: list[ManifestEntry], outputs: _LiveOutputs,
              staged: StagedOutputs, republished: bool) -> PublishReport:
    """Install the staged outputs, write the release record (unless identical to the one out), report."""
    rows = outputs.dedup.kept
    members = {m.guideline_id: m for m in deps.store.members(snap.id)}
    contributing = {r.doi for r in rows if r.edition == Edition.CURRENT}
    n_excluded = sum(1 for m in members.values() if m.excluded_reason)
    n_superseded = len({r.doi for r in rows if r.edition == Edition.SUPERSEDED})

    content = _release_content(deps, snap, entries, outputs, staged.checksums(), n_excluded)
    release_id, outcome = write_release(content, deps.publish_dir, deps.release_archive_dir, deps.store.backup,
                                        staged.install)
    return PublishReport(snap.id, republished, deps.publish_dir, len(contributing), n_excluded, n_superseded,
                         outputs.bundle.model, staged.paths, len(outputs.dedup.removed), release_id, outcome)


def _release_content(deps: PipelineDeps, snap: Snapshot, entries: list[ManifestEntry], outputs: _LiveOutputs,
                     checksums: dict[str, str], n_excluded: int) -> dict:
    """Everything the release id hashes (M5 spec §5): source, versions, inputs, counts, output checksums."""
    version = deps.store.get_extractor_version(snap.extractor_version_id)
    runs = _active_runs_by_guideline(deps.store, snap.id)
    validation = deps.store.latest_validation(snap.id)
    n_current = len(current_editions(entries))
    window = load_search_window(deps.search_path) if deps.search_path else None
    return {
        "snapshot": {"id": snap.id, "state": snap.state.value, "published_at": snap.published_at,
                     "accept_reason": snap.accept_reason},
        "extractor": {"version_id": version.id, "model_name": version.model_name,
                      "model_digest": version.model_digest, "embedding_revision": version.embedding_revision,
                      "classifier_sha256": version.classifier_sha256},
        "code": {"analysis_sha": deps.code_sha(),
                 "extraction_shas": sorted({run.code_sha for _, run in runs if run.code_sha})},
        "search": {"start": window.start.isoformat(), "cutoff": window.cutoff.isoformat()} if window else None,
        "edition_rule": EDITION_RULE,
        "validation": ({"gate": validation.gate.value, "baseline_snapshot_id": validation.baseline_snapshot_id}
                       if validation else None),
        "inputs": _release_inputs(deps, runs),
        "counts": {
            "guidelines": len(entries), "current": n_current, "superseded": len(entries) - n_current,
            "excluded": n_excluded, "recommendations": len(outputs.all_rows),
            "recommendations_kept": len(outputs.dedup.kept), "duplicates_removed": len(outputs.dedup.removed),
            "not_yet_assigned": _n_unassigned(deps, outputs.dedup.kept),
        },
        "outputs": checksums,
    }


def _release_inputs(deps: PipelineDeps, runs: list[tuple[Guideline, ExtractionRun]]) -> dict:
    """Hashes of every curated input the outputs derive from; strict status compares them with the release."""
    return {
        "manifest_sha256": file_sha256(deps.manifest_path),
        "candidates_sha256": _sha_or_none(deps.candidates_path),
        "search_sha256": _sha_or_none(deps.search_path),
        "dedup_verdicts_sha256": _sha_or_none(deps.dedup_verdicts_path),
        "themes_sha256": _sha_or_none(deps.themes_path),
        "pdfs": {g.doi: run.pdf_sha256 for g, run in runs},
        "runs": {g.doi: run.id for g, run in runs},
    }


def _stale_inputs(deps: PipelineDeps, runs: list[tuple[Guideline, ExtractionRun]], release: dict) -> list[str]:
    """Inputs changed since the release, e.g. ["search", "PDF of 10.1000/a"]; code commits do not count.

    A PDF counts only while its run is the released one: a new run is reported as outputs predating the runs.
    """
    now, then = _release_inputs(deps, runs), release["inputs"]
    stale = [key[:-len(_SHA_SUFFIX)] for key in sorted(now)
             if key.endswith(_SHA_SUFFIX) and now[key] != then.get(key)]
    stale += [f"PDF of {doi}" for doi, sha in sorted(now["pdfs"].items())
              if then["runs"].get(doi) == now["runs"][doi] and then["pdfs"].get(doi) != sha]
    return stale


def _live_verdicts(rows: list[HarmonizedRow],
                   deps: PipelineDeps) -> tuple[list[Adjudication], list[CandidatePair]]:
    """(verdicts bound to the screened pairs' texts, pairs without one); no ledger → no screen, nothing removed."""
    if deps.dedup_verdicts_path is None:
        return [], []
    return resolve(_screen_rows(rows, deps.similarity_model), load_adjudications(deps.dedup_verdicts_path))


def _gate_problems(deps: PipelineDeps, snap: Snapshot, entries: list[ManifestEntry]) -> list[str]:
    """M5 spec §5 gates 1, 3 and 5; coverage (2) and duplicate verdicts (4) are checked by the caller."""
    problems = _candidate_problems(deps, entries)
    problems += [f"{doi}: PDF missing or changed since its run (restore it, rebind-pdf, or add --force)"
                 for doi in _pdf_changed(deps.store, snap.id, _guidelines_by_id(deps.store))]
    try:
        _config_checked(deps, deps.store.get_extractor_version(snap.extractor_version_id))
    except ArtifactMismatchError as e:
        problems.append(f"extractor artefact drift: {e}")
    if deps.search_path is not None:
        try:
            load_search_window(deps.search_path)
        except ManifestError as e:
            problems += e.problems
    return problems


def _candidate_problems(deps: PipelineDeps, entries: list[ManifestEntry]) -> list[str]:
    """Included candidates and manifest rows differ: a curation step is missing."""
    if deps.candidates_path is None:
        return []
    try:
        check_candidates(load_candidates(deps.candidates_path), entries)
    except ManifestError as e:
        return e.problems
    return []


def _edition_problems(entries: list[ManifestEntry]) -> list[str]:
    """Current editions undecidable, e.g. an older edition no row supersedes."""
    try:
        current_editions(entries)
    except ManifestError as e:
        return e.problems
    return []


def _checked(deps: PipelineDeps, snap: Snapshot, entries: list[ManifestEntry], release: Optional[dict]) -> dict:
    """StatusReport fields of a strict check: every blocker publish would raise, plus stale public outputs."""
    blockers = [f"{doi}: no active run" for doi in _coverage_gaps(deps.store, snap.id, entries)]
    blockers += _gate_problems(deps, snap, entries)

    rows, _ = _harmonized_rows(deps.store, snap, entries)
    try:
        adjudications, unresolved = _live_verdicts(rows, deps)
    except DedupVerdictError as e:
        return {"blockers": blockers + e.problems}
    if unresolved:
        blockers.append(f"{len(unresolved)} screened duplicate pair(s) without a verdict for their texts")

    # The public outputs must be those of the active runs, e.g. not before an `add --no-publish`
    runs = {g.doi: run.id for g, run in _active_runs_by_guideline(deps.store, snap.id)}
    if release is None:
        blockers.append("no release record: publish first")
    elif snap.state == SnapshotState.PUBLISHED and (release["snapshot"]["id"] != snap.id
                                                    or release["inputs"]["runs"] != runs):
        blockers.append(f"public outputs predate the active runs: publish --snapshot {snap.id}")
    if release is not None:
        stale = _stale_inputs(deps, _active_runs_by_guideline(deps.store, snap.id), release)
        if stale:
            blockers.append(f"changed since release {release['release_id'][:_DIGEST_PREFIX]}: {', '.join(stale)}; "
                            f"publish --snapshot {snap.id}")

        # e.g. publish --package swapped the M4 outputs in, or a crash came between swap and release
        blockers += [f"{rel}: differs from release {release['release_id'][:_DIGEST_PREFIX]}; publish again"
                     for rel, sha in sorted(release["outputs"].items())
                     if _sha_or_none(os.path.join(deps.publish_dir, rel)) != sha]

    kept = deduplicate(rows, adjudications).kept if not unresolved else rows
    return {"blockers": blockers, "n_unresolved_pairs": len(unresolved), "n_unassigned": _n_unassigned(deps, kept)}


def _n_unassigned(deps: PipelineDeps, rows: list[HarmonizedRow]) -> Optional[int]:
    """Current recommendations the theme figure shows as "Not yet assigned"; None without a themes file."""
    if deps.themes_path is None:
        return None
    assignments = load_theme_assignments(deps.themes_path)
    return sum(1 for r in rows
               if r.edition == Edition.CURRENT and assignments.get(rec_key(r.doi, r.text), UNASSIGNED) == UNASSIGNED)


def _search_cutoff(deps: PipelineDeps) -> Optional[str]:
    return load_search_window(deps.search_path).cutoff.isoformat() if deps.search_path else None


def _prompt_mismatch(rebuilt: dict[int, str], calls: list[LlmCall]) -> Optional[str]:
    """First difference between the rebuilt prompts and the run's calls, e.g. "chunk 3 prompt differs"; None = same."""
    stored: dict[int, set[str]] = {}
    for c in calls:
        stored.setdefault(c.chunk_id, set()).add(c.prompt_sha256)  # self-consistency: one prompt, N samples
    if set(rebuilt) != set(stored):
        return f"{len(rebuilt)} chunks now, {len(stored)} chunks in the run"
    for chunk in sorted(rebuilt):
        if stored[chunk] != {rebuilt[chunk]}:
            return f"chunk {chunk} prompt differs"
    return None


def _log_rebind(path: Optional[str], doi: str, run_id: int, old_sha: Optional[str], new_sha: str) -> None:
    """Append one row to the rebind log; the DB keeps only the new hash."""
    if path is None:
        return
    is_new = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        if is_new:
            w.writerow(_REBIND_LOG_COLUMNS)
        w.writerow([doi, run_id, old_sha or "", new_sha, datetime.now(timezone.utc).isoformat(timespec="seconds")])


def _kept_recs_by_doi(store: Store, snapshot_id: int, removed: dict[tuple[str, int], int]) -> dict:
    """Active recommendations per DOI without the removed copies, for scoring."""
    return {g.doi: [r for r in recs if (g.doi, r.ordinal) not in removed]
            for g, recs in _active_recs_by_guideline(store, snapshot_id)}


def _rescored_development(deps: PipelineDeps, recs_by_doi: dict, similarity) -> dict:
    """The development report recomputed on the kept rows; the stored gate decision is not changed."""
    return json.loads(gate(score(recs_by_doi, deps.labelled(), similarity), None).to_json())


def _harmonized_rows(store: Store, snap: Snapshot,
                     entries: list[ManifestEntry]) -> tuple[list[HarmonizedRow], list[GuidelineFacts]]:
    """Rows from active runs of every included member, all editions; analytics decides which count where."""
    guidelines = {g.doi: g for g in store.list_guidelines()}
    current = current_editions(entries)
    members = {m.guideline_id: m for m in store.members(snap.id)}
    active = _active_recs(store, snap.id)

    rows = []
    facts = []
    for e in entries:
        g = guidelines[e.doi]
        edition = Edition.CURRENT if e.doi in current else Edition.SUPERSEDED
        member = members.get(g.id)
        facts.append(GuidelineFacts(e.doi, e.society, e.year, e.title, e.topic_id, edition, e.joint_with,
                                    member.excluded_reason if member else None))
        rows.extend(
            HarmonizedRow(g.doi, g.society, g.year,
                          harmonize(r.raw_strength, r.raw_certainty, r.text, r.raw_category, g.grading_family),
                          edition, r.text, r.page, r.raw_strength, r.raw_certainty, r.ordinal)
            for r in active.get(g.id, []))
    return rows, facts


def _screen_rows(rows: list[HarmonizedRow], similarity) -> list[CandidatePair]:
    """Duplicate screen per guideline: BioLORD cosine matrix of its rows, then evident.dedup.screen."""
    by_doi: dict[str, list[HarmonizedRow]] = {}
    for r in rows:
        by_doi.setdefault(r.doi, []).append(r)

    pairs = []
    for doi, group in sorted(by_doi.items()):
        if len(group) < 2:
            continue
        texts = [r.text for r in group]
        pairs.extend(screen(doi, [r.ordinal for r in group], texts, build_similarity_matrix(texts, texts, similarity)))
    return pairs


def _deduplicate(rows: list[HarmonizedRow], similarity,
                 cfg: AnalysisConfig) -> tuple[DedupResult, Optional[dict]]:
    """Frozen verdicts applied to the rows, with the Methods counts; no verdicts configured → rows unchanged.

    Raises DedupVerdictError unless the verdicts cover exactly the pairs the screen finds now.
    """
    if cfg.dedup_verdicts_path is None:
        return DedupResult(list(rows), {}, 0, {}), None

    pairs = _screen_rows(rows, similarity)
    adjudications = load_adjudications(cfg.dedup_verdicts_path)
    check_complete(pairs, adjudications)
    result = deduplicate(rows, adjudications)

    by_key = {(r.doi, r.ordinal): r for r in rows}
    removed = [by_key[key] for key in result.removed]
    facts = {
        "screen_biolord": SCREEN_BIOLORD,
        "screen_lexical": SCREEN_LEXICAL,
        "n_screened_pairs": len(pairs),
        "n_duplicate_pairs": sum(a.verdict == Verdict.DUPLICATE for a in adjudications),
        "n_removed": len(removed),
        "n_removed_current_graded": sum(r.edition == Edition.CURRENT and r.grade.category == Category.GRADED
                                        for r in removed),
        "removed_by_society": result.removed_by_society,
        "n_grade_conflicts": result.n_grade_conflicts,
        "adjudication_models": sorted({m for a in adjudications for m in a.model.split(MODEL_SEPARATOR)}),
        "human_check": _human_check(rows, result, cfg),
        "n_judges_split": sum(a.reason == SPLIT_REASON for a in adjudications),
        "n_blocked_by_grade": result.n_blocked_by_grade,
    }
    return result, facts


def _model_specification(store: Store, snap: Snapshot, version: ExtractorVersion) -> dict:
    """Supplement: the extraction prompt, each distinct few-shot set with the guidelines it served, the judge prompt.

    The prompt is shown with the first set; runs drew their examples per guideline (target excluded).
    """
    by_set: dict[str, list[str]] = {}
    for g, run in _active_runs_by_guideline(store, snap.id):
        by_set.setdefault(run.few_shot_json or "[]", []).append(g.doi)
    sets = [{"examples": json.loads(k), "dois": dois} for k, dois in sorted(by_set.items())]

    cfg = ExtractorConfig.from_json(version.config_json)
    examples = sets[0]["examples"] if sets else []
    return {"extraction_prompt": prompt_template(cfg, examples), "few_shot_sets": sets, "judge_prompt": JUDGE_PROMPT}


def _human_check(rows: list[HarmonizedRow], result: DedupResult, cfg: AnalysisConfig) -> Optional[dict]:
    """The reviewer's labelled pairs scored against the final clusters; None when no check is configured."""
    if cfg.dedup_human_sample_path is None:
        return None
    labels = load_human_check(cfg.dedup_human_sample_path, cfg.dedup_human_key_path)
    return asdict(score_human_check(labels, rows, result.removed))


def _bundle(deps: PipelineDeps, snap: Snapshot, rows: list[HarmonizedRow], facts: list[GuidelineFacts],
            paper: Optional[PaperPackage] = None) -> PublishBundle:
    model = _model_label(deps.store, snap)
    labelled_dois = {lg.doi for lg in deps.labelled()}
    return PublishBundle(
        snapshot_id=snap.id,
        extractor_version_id=snap.extractor_version_id,
        model=model,
        code_sha=deps.code_sha(),
        validation=_validation_summary(deps.store, snap.id),
        headline=headline(rows, snap.id, snap.extractor_version_id, model),
        certainty=certainty_distribution(rows),
        crosstab=strength_by_certainty(rows),
        trends=trends(rows),
        guidelines=guideline_summary(rows, facts),
        captions=caption_facts(rows, facts, labelled_dois),
        recs=rows,
        paper=paper,
        themes=theme_certainty(rows, load_theme_assignments(deps.themes_path)) if deps.themes_path else [],
    )


# ── package internals ───────────────────────────────────────────

def _audit_package(store: Store, snapshot_id: int, manifest_path: str, entries: list[ManifestEntry],
                   candidates: list[Candidate], inputs: PackageInputs) -> None:
    """Fail closed unless the source is exactly the frozen one the config names (spec §3)."""
    cfg = inputs.config
    snap = store.get_snapshot(snapshot_id)
    if snap is None or snap.state != SnapshotState.PUBLISHED:
        raise PackageAuditError([f"snapshot {snapshot_id} is {snap.state.value if snap else 'missing'}, "
                                 "not published"])

    problems = []
    if inputs.tree_state != TreeState.CLEAN:
        problems.append("analysis code has uncommitted changes; commit them so analysis_sha names the code")
    target = package_target_problem(inputs.package_dir)
    if target:
        problems.append(target)
    if os.path.exists(inputs.db_path + _WAL_SUFFIX):
        problems.append(f"{inputs.db_path}{_WAL_SUFFIX} exists: its frames are outside the DB hash")
    if snapshot_id != cfg.snapshot_id:
        problems.append(f"snapshot {snapshot_id} requested, config {cfg.name} names {cfg.snapshot_id}")
    for label, path, expected in (("database", inputs.db_path, cfg.db_sha256),
                                  ("manifest", manifest_path, cfg.manifest_sha256),
                                  ("candidates", cfg.candidates_path, cfg.candidates_sha256)):
        actual = _sha_or_none(path)
        if actual != expected:
            problems.append(f"{label} {path} SHA-256 {actual} differs from {expected}")

    if cfg.dedup_verdicts_path is not None:
        actual = _sha_or_none(cfg.dedup_verdicts_path)
        if actual != cfg.dedup_verdicts_sha256:
            problems.append(f"dedup verdicts {cfg.dedup_verdicts_path} SHA-256 {actual} differs from "
                            f"{cfg.dedup_verdicts_sha256}")

    for label, path, expected in (("human check sample", cfg.dedup_human_sample_path, cfg.dedup_human_sample_sha256),
                                  ("human check key", cfg.dedup_human_key_path, cfg.dedup_human_key_sha256)):
        if path is not None and _sha_or_none(path) != expected:
            problems.append(f"{label} {path} SHA-256 {_sha_or_none(path)} differs from {expected}")

    card_digest = (cfg.model_card or {}).get("model_digest")
    version_digest = store.get_extractor_version(snap.extractor_version_id).model_digest
    if cfg.model_card is not None and card_digest != version_digest:
        problems.append(f"model card describes digest {card_digest}, the runs used {version_digest}")

    if inputs.draft_template is not None and not os.path.isfile(inputs.draft_template):
        problems.append(f"draft template {inputs.draft_template} missing")
    if not os.path.isdir(cfg.spotcheck_dir):
        problems.append(f"spot-check dir {cfg.spotcheck_dir} missing")
    label_hashes = _dir_hashes(inputs.heldout_dir) if os.path.isdir(inputs.heldout_dir) else None
    if label_hashes is None:
        problems.append(f"held-out label dir {inputs.heldout_dir} missing")
    elif label_hashes != cfg.heldout_labels_sha256:
        problems.append(f"held-out labels in {inputs.heldout_dir} differ from the config's heldout_labels_sha256: "
                        f"{_hash_diff(label_hashes, cfg.heldout_labels_sha256)}")

    if len(entries) != cfg.n_guidelines:
        problems.append(f"{len(entries)} manifest guidelines, expected {cfg.n_guidelines}")
    try:
        n_current = len(current_editions(entries))
        if n_current != cfg.n_current:
            problems.append(f"{n_current} current editions, expected {cfg.n_current}")
    except ManifestError as e:
        problems += e.problems
    try:
        check_candidates(candidates, entries)
    except ManifestError as e:
        problems += e.problems

    manifest_dois = {e.doi for e in entries}
    problems += [f"held-out {d} is not a manifest guideline" for d in cfg.heldout_dois
                 if normalize_doi(d) not in manifest_dois]

    problems += _member_problems(store, snap.id, entries)
    if problems:
        raise PackageAuditError(problems)


def _member_problems(store: Store, snapshot_id: int, entries: list[ManifestEntry]) -> list[str]:
    """Every manifest guideline, and nothing else, has an active run that read the synced PDF bytes."""
    guidelines = _guidelines_by_id(store)
    members = store.members(snapshot_id)
    manifest_dois = {e.doi for e in entries}

    problems = [f"{doi}: no active run" for doi in _coverage_gaps(store, snapshot_id, entries)]
    for m in members:
        g = guidelines[m.guideline_id]
        if g.doi not in manifest_dois:
            problems.append(f"{g.doi}: snapshot member not in the manifest")
        if m.excluded_reason:
            problems.append(f"{g.doi}: excluded ({m.excluded_reason})")
        run = store.active_run(snapshot_id, m.guideline_id)
        if run is None:
            continue
        if not store.recommendations(run.id):
            problems.append(f"{g.doi}: run {run.id} has no recommendations")  # it would vanish from every table
        if run.pdf_sha256 is None:
            problems.append(f"{g.doi}: run {run.id} has no PDF hash")
        elif run.pdf_sha256 != g.pdf_sha256:
            problems.append(f"{g.doi}: run {run.id} read other PDF bytes than the guideline's")
    return problems


def _provenance(deps: PipelineDeps, snap: Snapshot, version: ExtractorVersion, inputs: PackageInputs) -> dict:
    """Sources, versions and checksums; analysis_manifest.json adds the output hashes."""
    cfg = inputs.config
    runs = [run for _, run in _active_runs(deps.store, snap.id)]
    return {
        "package": cfg.name,
        "snapshot": {"id": snap.id, "state": snap.state.value, "published_at": snap.published_at},
        "extractor": {"version_id": version.id, "model_name": version.model_name,
                      "model_digest": version.model_digest, "config": json.loads(version.config_json)},
        "code": {"analysis_sha": deps.code_sha(),
                 "extraction_shas": sorted({r.code_sha for r in runs if r.code_sha})},
        "search": {"start": cfg.search_start, "cutoff": cfg.search_cutoff},
        "edition_rule": EDITION_RULE,
        "bootstrap": {"n": cfg.bootstrap_n, "seed": cfg.bootstrap_seed, "unit": _BOOTSTRAP_UNIT},
        "analysis_config": asdict(cfg),
        "inputs": {
            "database": {"path": inputs.db_path, "sha256": file_sha256(inputs.db_path)},
            "manifest": {"path": deps.manifest_path, "sha256": file_sha256(deps.manifest_path)},
            "candidates": {"path": cfg.candidates_path, "sha256": file_sha256(cfg.candidates_path)},
            "analysis_config": {"path": inputs.config_path, "sha256": file_sha256(inputs.config_path)},
            "template": ({"path": inputs.draft_template, "sha256": file_sha256(inputs.draft_template)}
                         if inputs.draft_template else None),
            "heldout_labels": _dir_hashes(inputs.heldout_dir),
            "spotcheck": _dir_hashes(cfg.spotcheck_dir),
            "dedup_verdicts": ({"path": cfg.dedup_verdicts_path, "sha256": file_sha256(cfg.dedup_verdicts_path)}
                               if cfg.dedup_verdicts_path else None),
            "dedup_human_check": ({"sample": {"path": cfg.dedup_human_sample_path,
                                              "sha256": file_sha256(cfg.dedup_human_sample_path)},
                                   "key": {"path": cfg.dedup_human_key_path,
                                           "sha256": file_sha256(cfg.dedup_human_key_path)}}
                                  if cfg.dedup_human_sample_path else None),
            "pdfs": {g.doi: run.pdf_sha256 for g, run in _active_runs_by_guideline(deps.store, snap.id)},
        },
    }


def _dir_hashes(directory: str) -> dict[str, str]:
    """{file name: SHA-256} of the regular files in a directory, e.g. the held-out label set; lock files skipped."""
    names = sorted(f for f in os.listdir(directory)
                   if os.path.isfile(os.path.join(directory, f)) and not f.startswith(_IGNORED_FILE_PREFIXES))
    return {name: file_sha256(os.path.join(directory, name)) for name in names}


def _hash_diff(actual: dict[str, str], expected: dict[str, str]) -> str:
    """'changed a.xlsx; unexpected b.xlsx; missing c.xlsx'."""
    changed = sorted(n for n in actual.keys() & expected.keys() if actual[n] != expected[n])
    parts = [f"{label} {', '.join(names)}" for label, names in (
        ("changed", changed), ("unexpected", sorted(actual.keys() - expected.keys())),
        ("missing", sorted(expected.keys() - actual.keys()))) if names]
    return "; ".join(parts)


def _sha_or_none(path: str) -> Optional[str]:
    return file_sha256(path) if path and os.path.isfile(path) else None


def _active_runs(store: Store, snapshot_id: int) -> list[tuple[int, ExtractionRun]]:
    """[(guideline id, active run)] of included members that have one."""
    out = []
    for m in store.members(snapshot_id):
        run = None if m.excluded_reason else store.active_run(snapshot_id, m.guideline_id)
        if run:
            out.append((m.guideline_id, run))
    return out


def _active_runs_by_guideline(store: Store, snapshot_id: int) -> list[tuple[Guideline, ExtractionRun]]:
    guidelines = _guidelines_by_id(store)
    return sorted(((guidelines[gid], run) for gid, run in _active_runs(store, snapshot_id)),
                  key=lambda pair: pair[0].doi)


def _active_recs_by_guideline(store: Store, snapshot_id: int) -> list[tuple[Guideline, list[RawRecommendation]]]:
    return [(g, store.recommendations(run.id)) for g, run in _active_runs_by_guideline(store, snapshot_id)]


def _validation_summary(store: Store, snapshot_id: int) -> Optional[ValidationSummary]:
    validation = store.latest_validation(snapshot_id)
    if validation is None:
        return None
    return _summary_of(json.loads(validation.report_json), validation.gate, validation.baseline_snapshot_id)


def _summary_of(report: dict, gate: GateResult, baseline_snapshot_id: Optional[int]) -> Optional[ValidationSummary]:
    # A report without candidate metrics (hand-written in tests) shows as not validated
    candidate = report.get("candidate")
    if candidate is None:
        return None
    return ValidationSummary(
        gate=gate,
        n_guidelines=report["n_guidelines"],
        baseline_snapshot_id=baseline_snapshot_id,
        f1=candidate["f1"],
        f1_ci=tuple(candidate["f1_ci"]),
        precision=candidate["precision"],
        recall=candidate["recall"],
        combined_accuracy=candidate["combined_accuracy"],
        combined_ci=tuple(candidate["combined_ci"]),
    )
