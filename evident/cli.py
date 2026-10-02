"""Command line front end: python -m evident <command>.

    evident [--db PATH] [--manifest PATH] [--pdf-dir PATH] <command>
      snapshot --config CONFIG.json [--exclude DOI --reason TEXT]...
      snapshot --resume             [--exclude DOI --reason TEXT]...
      validate [--snapshot ID]
      publish  [--snapshot ID] [--accept-regression REASON]
      reject   --snapshot ID --reason TEXT      # retire a candidate for good
      add PDF  [--force]                       # then re-publish
      status

Maps flags to enums, builds the real PipelineDeps, prints reports.
Exit codes: 0 ok, 1 known error (message only), 2 usage error.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from typing import Optional

from evident import pipeline
from evident.corpus import ManifestError
from evident.extraction import ExtractionError, ExtractorConfig, Models, load_similarity_model, probe_artifacts, run
from evident.harmonization import UnsupportedGradingFamilyError
from evident.store import Store, StoreError
from evident.validation import ValidationJoinError, load_labelled
from extraction.llm_client import ModelNotFoundError

_DEFAULT_DB = "out/evident.sqlite"
_DEFAULT_MANIFEST = "corpus/manifest.csv"
_DEFAULT_PDF_DIR = "/mnt/data1/klug/datasets/evidence_extraction/pdfs"
_HEADLINE_PATH = "out/headline.csv"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_EXIT_OK = 0
_EXIT_ERROR = 1

_KNOWN_ERRORS = (pipeline.PipelineError, StoreError, ManifestError, ExtractionError, ValidationJoinError,
                 ModelNotFoundError, UnsupportedGradingFamilyError)


class _LazySimilarity:
    """BioLORD loaded on first use, once per process; status and publish never load it."""

    def __init__(self):
        self._model = None

    def __getattr__(self, name):
        if self._model is None:
            self._model = load_similarity_model()
        return getattr(self._model, name)


def main(argv: Optional[list[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "snapshot" and len(args.exclude) != len(args.reason):
        parser.error("each --exclude DOI needs one --reason TEXT")  # exits 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    with Store.open(args.db) as store:
        try:
            return _COMMANDS[args.command](args, _deps(args, store))
        except _KNOWN_ERRORS as e:
            print(f"error: {e}", file=sys.stderr)
            return _EXIT_ERROR


# ── commands ────────────────────────────────────────────────────

def _snapshot(args, deps) -> int:
    start = pipeline.SnapshotStart.RESUME if args.resume else pipeline.SnapshotStart.NEW
    config = _read_config(args.config) if args.config else None
    report = pipeline.snapshot(deps, start, config, list(zip(args.exclude, args.reason)))

    print(f"snapshot {report.snapshot_id} [{report.model}]: {report.state.value}")
    print(f"  succeeded {report.n_succeeded}, failed {report.n_failed}, empty {report.n_failed_empty}")
    for doi, error in report.failures:
        print(f"  failed {doi}: {error}")
    if report.pending:
        print(f"  pending {', '.join(report.pending)}  (fix, then: snapshot --resume [--exclude DOI --reason TEXT])")
    return _EXIT_OK


def _validate(args, deps) -> int:
    report = pipeline.validate(deps, args.snapshot)
    gate = report.gate
    baseline = f"{report.baseline_snapshot_id} [{report.baseline_model}]" if report.baseline_snapshot_id else "none"
    print(f"snapshot {report.snapshot_id} [{report.model}] vs baseline {baseline}: {gate.result.value}")
    _print_metrics(f"candidate {report.model}", gate.candidate)
    if gate.baseline:
        _print_metrics(f"baseline {report.baseline_model}", gate.baseline)
        print(f"  diff F1 {_fmt(gate.f1_diff)} {_fmt_ci(gate.f1_diff_ci)}, "
              f"combined {_fmt(gate.combined_diff)} {_fmt_ci(gate.combined_diff_ci)}")
    return _EXIT_OK


def _publish(args, deps) -> int:
    report = pipeline.publish(deps, args.accept_regression, args.snapshot)
    verb = "re-published" if report.republished else "published"
    print(f"snapshot {report.snapshot_id} [{report.model}] {verb}: {report.headline_path} "
          f"({report.n_headline_guidelines} guidelines; skipped {report.n_skipped_excluded} excluded, "
          f"{report.n_skipped_superseded} superseded)")
    return _EXIT_OK


def _add(args, deps) -> int:
    mode = pipeline.IngestMode.FORCE if args.force else pipeline.IngestMode.NORMAL
    report = pipeline.add(deps, args.pdf, mode)
    print(f"{report.doi}: {report.status.value} (snapshot {report.snapshot_id})")
    if report.status == pipeline.AddStatus.FAILED:
        print(f"error: {report.error}", file=sys.stderr)
        return _EXIT_ERROR

    # add = ingest, then re-publish that snapshot (never a pending candidate)
    return _publish(argparse.Namespace(accept_regression=None, snapshot=report.snapshot_id), deps)


def _reject(args, deps) -> int:
    report = pipeline.reject(deps, args.snapshot, args.reason)
    print(f"snapshot {report.snapshot_id} [{report.model}] rejected: {report.reason}")
    return _EXIT_OK


def _status(args, deps) -> int:
    r = pipeline.status(deps)
    if r.snapshot_id is None:
        print("no snapshot")
        return _EXIT_OK

    print(f"snapshot {r.snapshot_id}: {r.state.value}  model {r.model}  version {r.extractor_version_id[:12]}")
    print(f"  members {r.n_members}, excluded {len(r.excluded)}, pending {len(r.pending)}")
    print(f"  runs: succeeded {r.n_succeeded}, failed {r.n_failed}, empty {r.n_failed_empty}, "
          f"interrupted {r.n_interrupted}")
    if r.eta_s is not None and r.pending:
        print(f"  ETA {r.eta_s / 3600:.1f} h")
    print(f"  gate: {r.gate.value if r.gate else 'not validated'}")
    _print_list("excluded", [f"{d}: {reason}" for d, reason in r.excluded])
    _print_list("pending", r.pending)
    _print_list("failed runs", [f"{d}: {e}" for d, e in r.failures])
    _print_list("slow", [f"{d}: {s / 60:.0f} min" for d, s in r.slow])
    _print_list("outliers", [f"{d}: {why}" for d, why in r.outliers])
    _print_list("coverage gaps", r.coverage_gaps)
    return _EXIT_OK


_COMMANDS = {"snapshot": _snapshot, "validate": _validate, "publish": _publish, "reject": _reject,
             "add": _add, "status": _status}


# ── wiring ──────────────────────────────────────────────────────

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evident", description="Living evidence map pipeline.")
    parser.add_argument("--db", default=_DEFAULT_DB)
    parser.add_argument("--manifest", default=_DEFAULT_MANIFEST)
    parser.add_argument("--pdf-dir", default=_DEFAULT_PDF_DIR)
    commands = parser.add_subparsers(dest="command", required=True)

    snap = commands.add_parser("snapshot", help="extract every manifest guideline with one version")
    start = snap.add_mutually_exclusive_group(required=True)
    start.add_argument("--config", help="ExtractorConfig JSON for a new snapshot")
    start.add_argument("--resume", action="store_true", help="continue the building snapshot")
    snap.add_argument("--exclude", action="append", default=[], metavar="DOI")
    snap.add_argument("--reason", action="append", default=[], metavar="TEXT")

    val = commands.add_parser("validate", help="score against the labelled guidelines and gate")
    val.add_argument("--snapshot", type=int, help="snapshot id (default: latest complete)")

    pub = commands.add_parser("publish", help="publish the validated snapshot, write outputs")
    pub.add_argument("--snapshot", type=int, help="snapshot id (needed when several are complete)")
    pub.add_argument("--accept-regression", metavar="REASON", help="publish despite a failed gate")

    rej = commands.add_parser("reject", help="retire a building or complete snapshot, with a reason")
    rej.add_argument("--snapshot", type=int, required=True)
    rej.add_argument("--reason", required=True)

    add = commands.add_parser("add", help="extract one new guideline with the published version, then publish")
    add.add_argument("pdf")
    add.add_argument("--force", action="store_true", help="extract again even if already present")

    commands.add_parser("status", help="progress, ETA, failures, outliers, coverage")
    return parser


def _deps(args, store: Store) -> pipeline.PipelineDeps:
    similarity = _LazySimilarity()
    return pipeline.PipelineDeps(
        store=store,
        manifest_path=args.manifest,
        pdf_dir=args.pdf_dir,
        runner=lambda cfg, pdf, doi: run(cfg, pdf, doi, Models(similarity=similarity)),
        probe=probe_artifacts,
        labelled=load_labelled,
        similarity_model=similarity,
        code_sha=_git_head,
        headline_path=_HEADLINE_PATH,
    )


def _read_config(path: str) -> ExtractorConfig:
    with open(path, encoding="utf-8") as f:
        return ExtractorConfig.from_json(f.read())


def _git_head() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_ci(ci) -> str:
    return "" if not ci else f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"


def _print_metrics(label: str, m) -> None:
    print(f"  {label}: F1 {_fmt(m.f1)} {_fmt_ci(m.f1_ci)}  P {_fmt(m.precision)}  R {_fmt(m.recall)}")
    print(f"    strength {_fmt(m.strength_accuracy)}  certainty {_fmt(m.certainty_accuracy)}  "
          f"combined {_fmt(m.combined_accuracy)} {_fmt_ci(m.combined_ci)}")
    print(f"    ungraded P {_fmt(m.ungraded_precision)}  R {_fmt(m.ungraded_recall)}")


def _print_list(title: str, items: list[str]) -> None:
    if not items:
        return
    print(f"  {title}:")
    for item in items:
        print(f"    {item}")
