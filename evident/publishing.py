"""Render one published snapshot: CSV tables, figures, captions and a static dashboard.

    PublishBundle (aggregates from analytics, built by the pipeline)
        │ render(bundle, out_dir)
        ▼
    <out_dir>/.publish-XXXX/ ── tables/*.csv, figures/*.svg|png + captions.md, dashboard/index.html
        │ all written? swap each subdir in; else delete the staging dir (previous outputs stay)
        ▼
    <out_dir>/tables  <out_dir>/figures  <out_dir>/dashboard

    bundle.paper (optional) ── render_package(bundle, package_dir) ──▶ <package_dir>/ swapped in whole:
        CSVs (1 decimal, NA), JSON (unrounded), captions.md, draft.md (template, unknown field = error),
        analysis_manifest.json (inputs + hash of every other emitted file)

Paper figures and dashboard share the same files (design Decision 4). Output is
deterministic: same bundle → byte-identical files (fixed SVG hash salt, no dates).
Never reads the store; only renders what it is given.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from typing import Optional

import matplotlib
from jinja2 import Environment, FileSystemLoader, StrictUndefined, UndefinedError, select_autoescape
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from evident.analytics import (
    CaptionFacts,
    CertaintyRow,
    CrosstabRow,
    GuidelineRow,
    HarmonizedRow,
    HeadlineRow,
    TrendRow,
)
from evident.domain import Certainty, GateResult, Strength
from evident.paper import BiasRow, PaperPackage, Proportion

_TABLES = "tables"
_FIGURES = "figures"
_DASHBOARD = "dashboard"
_SUBDIRS = (_TABLES, _FIGURES, _DASHBOARD)
_STAGING_PREFIX = ".publish-"
_OLD_SUFFIX = ".old"
_CAPTIONS_FILE = "captions.md"
_DASHBOARD_FILE = "index.html"
_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
_TEMPLATE = "dashboard.html.j2"

_ALL = "ALL"
_PNG_DPI = 300
_SVG_HASH_SALT = "evident"  # fixed ids in SVG output: same figure → same bytes
_VERSION_PREFIX = 12
_SHA_PREFIX = 7

_PACKAGE_STAGING_PREFIX = ".package-"
_NA = "NA"  # an empty denominator, never 0
_MANIFEST_FILE = "analysis_manifest.json"
_DRAFT_FILE = "draft.md"
_HELDOUT_NOTE = "n={n} guidelines: imprecise; reported separately, not part of the development gate"
_OPTIONAL_NESTED = {(BiasRow, "correct"): Proportion}  # Optional dataclass fields, for stable CSV columns
_DEVELOPMENT_NOTE = ("the labelled guidelines also informed extractor development and tuning: "
                     "in-sample performance, not an external test")

# One colour per society everywhere (figures and dashboard); ALL in neutral grey
_SOCIETY_COLOURS = {"ACP": "#1f77b4", "ESICM": "#d62728", "SCCM": "#2ca02c", _ALL: "#555555"}
_FALLBACK_COLOUR = "#9467bd"
_CERTAINTY_COLOURS = {Certainty.HIGH: "#08519c", Certainty.MODERATE: "#4292c6",
                      Certainty.LOW: "#9ecae1", Certainty.VERY_LOW: "#deebf7"}
_CERTAINTY_LABELS = {Certainty.HIGH: "High", Certainty.MODERATE: "Moderate",
                     Certainty.LOW: "Low", Certainty.VERY_LOW: "Very low"}
_STRENGTH_LABELS = {Strength.STRONG: "Strong", Strength.WEAK: "Weak / conditional"}


@dataclass(frozen=True)
class ValidationSummary:
    """Latest validation of the snapshot, as shown in captions and the dashboard."""
    gate: GateResult
    n_guidelines: int
    baseline_snapshot_id: Optional[int]
    f1: Optional[float]
    f1_ci: tuple[Optional[float], Optional[float]]
    precision: Optional[float]
    recall: Optional[float]
    combined_accuracy: Optional[float]
    combined_ci: tuple[Optional[float], Optional[float]]


@dataclass(frozen=True)
class PublishBundle:
    """Everything one publish renders; built by the pipeline from analytics."""
    snapshot_id: int
    extractor_version_id: str
    model: str
    code_sha: Optional[str]
    validation: Optional[ValidationSummary]
    headline: list[HeadlineRow]
    certainty: list[CertaintyRow]
    crosstab: list[CrosstabRow]
    trends: list[TrendRow]
    guidelines: list[GuidelineRow]
    captions: CaptionFacts
    recs: list[HarmonizedRow]  # current and superseded editions, for the browser
    paper: Optional[PaperPackage] = None  # only with publish --package


class DraftFieldError(ValueError):
    """The draft template names a field the package does not export; no number is typed by hand."""


@dataclass(frozen=True)
class _Figure:
    id: str
    title: str
    caption: str


def render(bundle: PublishBundle, out_dir: str) -> list[str]:
    """Write every output; returns the final paths. A failure leaves the previous outputs intact."""
    os.makedirs(out_dir, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=_STAGING_PREFIX, dir=out_dir)
    try:
        _write_tables(bundle, os.path.join(staging, _TABLES))
        figures = _write_figures(bundle, os.path.join(staging, _FIGURES))
        _write_dashboard(bundle, figures, os.path.join(staging, _FIGURES), os.path.join(staging, _DASHBOARD))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    written = sorted(os.path.join(out_dir, os.path.relpath(os.path.join(root, f), staging))
                     for root, _, files in os.walk(staging) for f in files)
    _swap_in(staging, out_dir)
    return written


# ── tables ──────────────────────────────────────────────────────

def _write_tables(bundle: PublishBundle, directory: str) -> None:
    os.makedirs(directory)
    _write_csv(bundle.headline, os.path.join(directory, "headline.csv"))
    _write_csv(bundle.certainty, os.path.join(directory, "certainty.csv"))
    _write_csv(bundle.crosstab, os.path.join(directory, "strength_by_certainty.csv"))
    _write_csv(bundle.trends, os.path.join(directory, "trends.csv"))
    _write_csv(bundle.guidelines, os.path.join(directory, "guidelines.csv"))
    _write_dicts([_rec_dict(r) for r in bundle.recs], os.path.join(directory, "recommendations.csv"))


def _write_csv(rows: list, path: str) -> None:
    _write_dicts([{k: _cell(v) for k, v in asdict(r).items()} for r in rows], path,
                 [f.name for f in fields(rows[0])] if rows else [])


def _write_dicts(rows: list[dict], path: str, columns: Optional[list[str]] = None) -> None:
    columns = columns if columns is not None else (list(rows[0]) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _cell(value):
    """CSV cell: empty for None, enum value, ';'-joined tuple."""
    if value is None:
        return ""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return ";".join(str(v) for v in value)
    return value


def _rec_dict(r: HarmonizedRow) -> dict:
    g = r.grade
    return {
        "doi": r.doi, "society": r.society, "year": r.year, "edition": r.edition.value,
        "page": _cell(r.page), "text": r.text, "raw_strength": r.raw_strength, "raw_certainty": r.raw_certainty,
        "category": g.category.value, "strength": _cell(g.strength), "direction": _cell(g.direction),
        "certainty": _cell(g.certainty), "strength_status": g.strength_status.value,
        "certainty_status": g.certainty_status.value,
    }


# ── captions ────────────────────────────────────────────────────

def _figures(bundle: PublishBundle) -> list[_Figure]:
    """Short, distinct captions; the common corpus/provenance note is stated once, beside them."""
    c = bundle.captions
    trend_note = (f"All editions: {c.n_recs_all_editions} recommendations, including "
                  f"{c.n_superseded_guidelines} superseded edition(s). Few guidelines per society and year; "
                  "n graded is shown per point; the last year is incomplete.")
    return [
        _Figure("fig1_certainty_by_society", "Certainty of evidence by society",
                "Share of current graded recommendations at each certainty level; n = certainty mapped."),
        _Figure("fig2_strong_by_society", "Strong recommendations by society",
                "Share of current graded recommendations that are strong (for or against); "
                "n = strength mapped."),
        _Figure("fig3_strength_by_certainty", "Strength by certainty",
                "Counts of current graded recommendations by strength and certainty (both mapped)."),
        _Figure("fig4_trends", "Trends by publication year",
                f"Share of strong and of high/moderate-certainty recommendations per first-online year. "
                f"{trend_note}"),
    ]


def _common_caption(bundle: PublishBundle) -> str:
    """Corpus, denominator, exclusions, joint guidelines, validation flags and provenance."""
    c = bundle.captions
    years = f"{c.year_min}–{c.year_max}" if c.year_min is not None else "no years"
    parts = [
        f"Current editions of {c.n_guidelines} {_join(c.societies)} guidelines ({years}): "
        f"{c.n_recs} recommendations, {c.n_graded} graded.",
        f"Percentages are over graded recommendations with the axis mapped; not in the denominators: "
        f"{c.n_strength_unmapped} unmapped strength, {c.n_certainty_unmapped} unmapped certainty, "
        f"{c.n_certainty_ungraded} graded without a certainty grade.",
        _excluded_sentence(c),
    ]
    if c.joint:
        joint = "; ".join(f"{doi} ({society} with {_join(others)})" for doi, society, others in c.joint)
        parts.append(f"Joint guidelines are counted once, under the society listed: {joint}.")
    parts += _validation_notes(bundle)
    parts.append(_provenance(bundle))
    return " ".join(parts)


def _validation_notes(bundle: PublishBundle) -> list[str]:
    """Per society without development labels: unvalidated, or evaluated on the held-out set (package only)."""
    heldout = bundle.paper.heldout_societies if bundle.paper else ()
    notes = []
    for s in bundle.captions.unvalidated_societies:
        if s not in heldout:
            notes.append(f"{s}: extraction accuracy unvalidated (no labelled {s} guideline).")
            continue
        n_dev = bundle.validation.n_guidelines if bundle.validation else 0
        notes.append(f"{s}: evaluated separately on {bundle.paper.validation_heldout.n_guidelines} held-out "
                     f"guidelines; not part of the {n_dev}-guideline development gate.")
    return notes


def _excluded_sentence(c: CaptionFacts) -> str:
    if not c.excluded:
        return "No guideline excluded."
    listed = "; ".join(f"{doi} ({reason})" for doi, reason in c.excluded)
    return f"{len(c.excluded)} guideline(s) excluded: {listed}."


def _provenance(bundle: PublishBundle) -> str:
    text = (f"Snapshot {bundle.snapshot_id}, extractor version {bundle.extractor_version_id[:_VERSION_PREFIX]}, "
            f"model {bundle.model}.")
    v = bundle.validation
    if v is None:
        return text + " Not validated."
    return (text + f" Validation on {v.n_guidelines} labelled guidelines: graded F1 {_num(v.f1)} {_ci(v.f1_ci)}, "
            f"combined grade accuracy {_num(v.combined_accuracy)} {_ci(v.combined_ci)}.")


def _write_captions(bundle: PublishBundle, figures: list[_Figure], path: str) -> None:
    lines = ["# Captions", ""]
    for i, fig in enumerate(figures, start=1):
        lines += [f"**Figure {i}. {fig.title}.** {fig.caption}", ""]
    lines += ["**Table 1. Headline by society.** Current editions; one row per society and all.", "",
              f"**Notes (all figures and Table 1).** {_common_caption(bundle)}", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── figures ─────────────────────────────────────────────────────

def _write_figures(bundle: PublishBundle, directory: str) -> list[_Figure]:
    os.makedirs(directory)
    figures = _figures(bundle)
    painters = {
        "fig1_certainty_by_society": _paint_certainty,
        "fig2_strong_by_society": _paint_strong,
        "fig3_strength_by_certainty": _paint_crosstab,
        "fig4_trends": _paint_trends,
    }
    with matplotlib.rc_context({"svg.hashsalt": _SVG_HASH_SALT, "svg.fonttype": "path", "font.size": 9}):
        for fig in figures:
            figure = painters[fig.id](bundle)
            FigureCanvasAgg(figure)
            figure.savefig(os.path.join(directory, fig.id + ".svg"), format="svg", metadata={"Date": None})
            figure.savefig(os.path.join(directory, fig.id + ".png"), format="png", dpi=_PNG_DPI,
                           metadata={"Software": None})
    _write_captions(bundle, figures, os.path.join(directory, _CAPTIONS_FILE))
    return figures


def _paint_certainty(bundle: PublishBundle) -> Figure:
    """Horizontal stacked bars: one per society + ALL, segments = certainty levels (%)."""
    societies = _societies(bundle.certainty)
    figure = Figure(figsize=(6.5, 0.6 * len(societies) + 1.2))
    ax = figure.add_subplot()
    for level in _CERTAINTY_COLOURS:
        lefts = [_left_of(bundle.certainty, s, level) for s in societies]
        widths = [_pct_or_zero(_find(bundle.certainty, s, level).pct) for s in societies]
        ax.barh(societies, widths, left=lefts, color=_CERTAINTY_COLOURS[level], label=_CERTAINTY_LABELS[level],
                edgecolor="white")
    for i, s in enumerate(societies):
        n = sum(r.n for r in bundle.certainty if r.society == s)
        ax.text(101, i, f"n={n}", va="center")
    ax.set_xlim(0, 112)
    ax.set_xlabel("% of graded recommendations")
    ax.invert_yaxis()
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.25), frameon=False)
    figure.tight_layout()
    return figure


def _paint_strong(bundle: PublishBundle) -> Figure:
    rows = bundle.headline
    figure = Figure(figsize=(5, 3.2))
    ax = figure.add_subplot()
    names = [r.society for r in rows]
    values = [_pct_or_zero(r.pct_strong) for r in rows]
    ax.bar(names, values, color=[_colour(n) for n in names])
    for i, r in enumerate(rows):
        label = "n/a" if r.pct_strong is None else f"{r.pct_strong:.1f}%"
        ax.text(i, values[i] + 1.5, f"{label}\nn={r.n_graded - r.n_strength_unmapped}", ha="center", va="bottom")
    ax.set_ylim(0, 115)
    ax.set_ylabel("% strong")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    figure.tight_layout()
    return figure


def _paint_crosstab(bundle: PublishBundle) -> Figure:
    """One heatmap panel per society + ALL: rows = strength, columns = certainty, cells = counts."""
    societies = _societies(bundle.crosstab)
    levels = list(_CERTAINTY_COLOURS)
    strengths = list(_STRENGTH_LABELS)
    figure = Figure(figsize=(2.3 * len(societies) + 0.8, 2.4))
    axes = figure.subplots(1, len(societies), squeeze=False)[0]
    for ax, society in zip(axes, societies):
        counts = [[_find_cross(bundle.crosstab, society, st, lv).n for lv in levels] for st in strengths]
        peak = max(max(row) for row in counts) or 1
        ax.imshow(counts, cmap="Blues", vmin=0, vmax=peak, aspect="auto")
        for i, row in enumerate(counts):
            for j, n in enumerate(row):
                ax.text(j, i, str(n), ha="center", va="center", color="white" if n > peak / 2 else "black")
        ax.set_title(society)
        ax.set_xticks(range(len(levels)), [_CERTAINTY_LABELS[lv] for lv in levels], rotation=45, ha="right")
        ax.set_yticks(range(len(strengths)), [_STRENGTH_LABELS[st] if ax is axes[0] else "" for st in strengths])
    figure.tight_layout()
    return figure


def _paint_trends(bundle: PublishBundle) -> Figure:
    """Two panels (% strong, % high/moderate) by year; one line per society, ALL dashed; n graded per point."""
    figure = Figure(figsize=(7.5, 3.2))
    left, right = figure.subplots(1, 2, sharex=True)
    for ax, attr, title in ((left, "pct_strong", "% strong"), (right, "pct_high_moderate", "% high/moderate")):
        for society in _societies(bundle.trends):
            rows = [r for r in bundle.trends if r.society == society]
            years = [r.year for r in rows]
            values = [math.nan if getattr(r, attr) is None else getattr(r, attr) for r in rows]
            ax.plot(years, values, marker="o", label=society, color=_colour(society),
                    linestyle="--" if society == _ALL else "-")
            if society == _ALL:
                continue
            for r, y in zip(rows, values):
                if not math.isnan(y):
                    ax.annotate(str(r.n_graded), (r.year, y), textcoords="offset points", xytext=(0, 4),
                                ha="center", fontsize=6, color=_colour(society))
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.xaxis.get_major_locator().set_params(integer=True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    if bundle.trends:
        left.legend(frameon=False, fontsize=7)
    figure.tight_layout()
    return figure


def _societies(rows) -> list[str]:
    """Society order of the rows (analytics: sorted names, then ALL)."""
    seen = []
    for r in rows:
        if r.society not in seen:
            seen.append(r.society)
    return seen


def _find(rows: list[CertaintyRow], society: str, level: Certainty) -> CertaintyRow:
    return next(r for r in rows if r.society == society and r.certainty == level)


def _find_cross(rows: list[CrosstabRow], society: str, strength: Strength, level: Certainty) -> CrosstabRow:
    return next(r for r in rows if r.society == society and r.strength == strength and r.certainty == level)


def _left_of(rows: list[CertaintyRow], society: str, level: Certainty) -> float:
    """Sum of the % of levels drawn before `level` (stacking offset)."""
    order = list(_CERTAINTY_COLOURS)
    return sum(_pct_or_zero(_find(rows, society, lv).pct) for lv in order[:order.index(level)])


def _pct_or_zero(pct: Optional[float]) -> float:
    return 0.0 if pct is None else pct


def _colour(society: str) -> str:
    return _SOCIETY_COLOURS.get(society, _FALLBACK_COLOUR)


# ── dashboard ───────────────────────────────────────────────────

def _write_dashboard(bundle: PublishBundle, figures: list[_Figure], figure_dir: str, directory: str) -> None:
    """Self-contained HTML: figures as SVG data URIs, data as embedded JSON; no external requests."""
    os.makedirs(directory)
    env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=select_autoescape(["html", "j2"]),
                      keep_trailing_newline=True)
    c = bundle.captions
    html = env.get_template(_TEMPLATE).render(
        snapshot_id=bundle.snapshot_id,
        version=bundle.extractor_version_id[:_VERSION_PREFIX],
        model=bundle.model,
        code_sha=(bundle.code_sha or "unknown")[:_SHA_PREFIX],
        validation=_validation_view(bundle.validation),
        validation_notes=_validation_notes(bundle),
        facts=c,
        headline=[{k: _cell(v) for k, v in asdict(r).items()} for r in bundle.headline],
        figures=[{"title": f.title, "caption": f.caption, "src": _svg_data_uri(os.path.join(figure_dir, f.id + ".svg"))}
                 for f in figures],
        common_caption=_common_caption(bundle),
        society_colours=_SOCIETY_COLOURS,
        data_json=_embedded_json({
            "guidelines": [{k: _cell(v) for k, v in asdict(r).items()} for r in bundle.guidelines],
            "recs": [_rec_dict(r) for r in bundle.recs],
        }),
    )
    with open(os.path.join(directory, _DASHBOARD_FILE), "w", encoding="utf-8") as f:
        f.write(html)


def _validation_view(v: Optional[ValidationSummary]) -> Optional[dict]:
    if v is None:
        return None
    return {"gate": v.gate.value, "n": v.n_guidelines, "f1": _num(v.f1), "f1_ci": _ci(v.f1_ci),
            "precision": _num(v.precision), "recall": _num(v.recall),
            "combined": _num(v.combined_accuracy), "combined_ci": _ci(v.combined_ci)}


def _svg_data_uri(path: str) -> str:
    """An <img> data URI, not inline SVG: matplotlib ids would collide across figures in one page."""
    with open(path, "rb") as f:
        return "data:image/svg+xml;base64," + base64.b64encode(f.read()).decode("ascii")


def _embedded_json(data: dict) -> str:
    """JSON safe inside <script>: every '<' escaped, so '</script>' in a text can't close the tag."""
    return json.dumps(data, sort_keys=True, ensure_ascii=False).replace("<", "\\u003c")


# ── analysis package ────────────────────────────────────────────

def render_package(bundle: PublishBundle, package_dir: str) -> list[str]:
    """Write the paper package; returns the final paths. A failure leaves the previous package intact."""
    if bundle.paper is None:
        raise ValueError("render_package needs a bundle with a paper payload")

    parent = os.path.dirname(os.path.abspath(package_dir))
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=_PACKAGE_STAGING_PREFIX, dir=parent)
    try:
        _write_package(bundle, staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    names = sorted(os.listdir(staging))
    _swap_dir(staging, os.path.abspath(package_dir))
    return [os.path.join(package_dir, n) for n in names]


def _write_package(bundle: PublishBundle, directory: str) -> None:
    p = bundle.paper
    _write_package_csv(p.table1, os.path.join(directory, "table1_society.csv"))
    _write_package_csv(p.table2, os.path.join(directory, "table2_certainty_strength.csv"))
    _write_package_csv(p.guideline_mean, os.path.join(directory, "table_s1_guideline_mean.csv"))
    _write_package_csv(p.figure_data, os.path.join(directory, "figure_data.csv"))
    _write_package_csv(p.corpus_flow, os.path.join(directory, "corpus_flow.csv"))
    _write_package_csv(p.excluded, os.path.join(directory, "excluded_guidelines.csv"))
    _write_package_csv(p.raw_mapping, os.path.join(directory, "raw_harmonized_mapping.csv"))
    _write_package_csv(p.spotcheck_bias, os.path.join(directory, "spotcheck_bias.csv"))

    exports = _package_json(bundle)
    for name, data in exports.items():
        _write_json(data, os.path.join(directory, name + ".json"))

    _write_package_captions(bundle, os.path.join(directory, "captions.md"))
    with open(os.path.join(directory, _DRAFT_FILE), "w", encoding="utf-8") as f:
        f.write(_render_draft(bundle, exports))

    # Last: hashes of every other file; the manifest never lists itself
    outputs = {name: _file_sha256(os.path.join(directory, name)) for name in sorted(os.listdir(directory))}
    _write_json({**p.provenance, "outputs": outputs}, os.path.join(directory, _MANIFEST_FILE))


def _package_json(bundle: PublishBundle) -> dict[str, dict]:
    """The package's JSON exports, by file stem; validation samples kept apart and labelled."""
    p = bundle.paper
    return {
        "validation_development": {
            "sample": "development", "in_sample": True, "gated": True, "note": _DEVELOPMENT_NOTE,
            "report": p.validation_development,
        },
        "validation_sccm_heldout": {
            "sample": "heldout", "in_sample": False, "gated": False,
            "note": _HELDOUT_NOTE.format(n=p.validation_heldout.n_guidelines),
            "dois": list(p.config.heldout_dois),
            "report": _plain(asdict(p.validation_heldout)),
        },
        "methods_facts": p.methods_facts,
    }


def _write_package_csv(rows: list, path: str) -> None:
    flat = [_flat(r) for r in rows]
    _write_dicts([{k: _package_cell(v) for k, v in r.items()} for r in flat], path,
                 list(_flat_names(rows[0])) if rows else [])


def _flat(row) -> dict:
    """Dataclass → flat dict, nested dataclasses prefixed: strong=Proportion(…) → strong_num, strong_pct, …

    An Optional nested field that is None (BiasRow.correct) still gets its columns, all None.
    """
    out = {}
    for f in fields(row):
        value = getattr(row, f.name)
        nested = _OPTIONAL_NESTED.get((type(row), f.name))
        if is_dataclass(value):
            out.update({f"{f.name}_{k}": v for k, v in _flat(value).items()})
        elif value is None and nested:
            out.update({f"{f.name}_{n.name}": None for n in fields(nested)})
        else:
            out[f.name] = value.value if isinstance(value, Enum) else value
    return out


def _flat_names(row) -> list[str]:
    return list(_flat(row))


def _package_cell(value):
    """CSV cell: NA for None, one decimal for a float (all package floats are percentages)."""
    if value is None:
        return _NA
    if isinstance(value, float):
        return f"{value:.1f}"
    return value


def _render_draft(bundle: PublishBundle, exports: dict[str, dict]) -> str:
    """Render the manuscript template; every number comes from a named export field."""
    p = bundle.paper
    template_dir, name = os.path.split(os.path.abspath(p.config.template_path))
    env = Environment(loader=FileSystemLoader(template_dir), undefined=StrictUndefined, keep_trailing_newline=True,
                      autoescape=False)
    env.filters.update(pct=_fmt_pct, ci=_fmt_ci_pct, p3=_fmt_p3, ci3=_fmt_ci3)
    try:
        return env.get_template(name).render(**_draft_context(bundle, exports))
    except UndefinedError as e:
        raise DraftFieldError(f"{p.config.template_path}: {e}") from e


def _draft_context(bundle: PublishBundle, exports: dict[str, dict]) -> dict:
    """Template namespace, keyed like the export files: table1_society.ACP.strong_pct is that CSV cell."""
    p = bundle.paper
    t2: dict = {}
    for r in p.table2:
        t2.setdefault(r.society, {}).setdefault(r.table, {})[r.certainty.value] = _flat(r)
    s1: dict = {}
    for r in p.guideline_mean:
        s1.setdefault(r.society, {})[r.metric.value] = _flat(r)
    flow: dict = {}
    for r in p.corpus_flow:
        flow.setdefault(r.stage, {})[r.item] = r.n
    return {
        "table1_society": {r.society: _flat(r) for r in p.table1},
        "table2_certainty_strength": t2,
        "table_s1_guideline_mean": s1,
        "corpus_flow": flow,
        "excluded_guidelines": [_flat(r) for r in p.excluded],
        "raw_harmonized_mapping": [_flat(r) for r in p.raw_mapping],
        "spotcheck_bias": {r.metric.value: _flat(r) for r in p.spotcheck_bias},
        "captions": {f.id: f.caption for f in _figures(bundle)},
        "notes": {"common": _common_caption(bundle)},
        "analysis": p.provenance,
        **exports,
    }


def _write_package_captions(bundle: PublishBundle, path: str) -> None:
    """Figure captions as in figures/captions.md, plus the package tables; common note once."""
    p = bundle.paper
    lines = ["# Captions", ""]
    for i, fig in enumerate(_figures(bundle), start=1):
        lines += [f"**Figure {i}. {fig.title}.** {fig.caption}", ""]
    lines += [
        "**Table 1. Graded recommendations by society.** Current editions. Percentages over graded "
        "recommendations with the axis mapped; n/N beside each. ALL is pooled per recommendation "
        "(SCCM-dominated), not a mean of society percentages. 95% intervals: guideline-cluster bootstrap "
        f"({p.config.bootstrap_n} resamples, seed {p.config.bootstrap_seed}), within-corpus variability between "
        "guidelines; they do not correct extraction error.", "",
        "**Table 2. Certainty, and strength by certainty.** Current editions; counts with row percentages.", "",
        "**Table S1. Guideline-weighted sensitivity.** Unweighted mean of each guideline's percentage, "
        "against the pooled estimate; guidelines with an empty denominator are left out and counted.", "",
        f"**Notes.** {_common_caption(bundle)}", "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_json(data, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_plain(data), sort_keys=True, indent=2, ensure_ascii=False) + "\n")


def _plain(value):
    """Enums → values, tuples → lists, recursively; JSON and templates see plain data."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _fmt_pct(value: Optional[float]) -> str:
    return _NA if value is None else f"{value:.1f}"


def _fmt_ci_pct(low: Optional[float], high: Optional[float]) -> str:
    """Journal style, safe for negative bounds: "-2.5 to 6.9"."""
    return _NA if low is None or high is None else f"{low:.1f} to {high:.1f}"


def _fmt_p3(value: Optional[float]) -> str:
    return _NA if value is None else f"{value:.3f}"


def _fmt_ci3(ci) -> str:
    return _NA if not ci or ci[0] is None or ci[1] is None else f"{ci[0]:.3f} to {ci[1]:.3f}"


# ── file swap ───────────────────────────────────────────────────

def _swap_in(staging: str, out_dir: str) -> None:
    """Replace each output subdir by its staged version, then drop the staging dir."""
    for sub in _SUBDIRS:
        _swap_dir(os.path.join(staging, sub), os.path.join(out_dir, sub))
    os.rmdir(staging)


def _swap_dir(staged: str, final: str) -> None:
    old = final + _OLD_SUFFIX
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(final):
        os.rename(final, old)
    os.rename(staged, final)
    shutil.rmtree(old, ignore_errors=True)


# ── formatting ──────────────────────────────────────────────────

def _num(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _ci(ci: tuple[Optional[float], Optional[float]]) -> str:
    if not ci or ci[0] is None or ci[1] is None:
        return ""
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _join(items) -> str:
    """("ACP", "ESICM", "SCCM") → "ACP, ESICM and SCCM"."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]
