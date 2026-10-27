"""Render one published snapshot: CSV tables, figures, captions and a static dashboard.

    PublishBundle (aggregates from analytics and themes, built by the pipeline)
        │ render(bundle, out_dir), or stage(bundle, out_dir) … StagedOutputs.install()
        ▼
    <out_dir>/.publish-XXXX/ ── tables/*.csv, figures/*.svg|png + captions.md, dashboard/index.html
        │ all written? swap each subdir in; else delete the staging dir (previous outputs stay)
        │ a failed swap puts every previous subdir back (all three or none)
        ▼
    <out_dir>/tables  <out_dir>/figures  <out_dir>/dashboard

    bundle.paper (optional) ── render_package(bundle, package_dir, out_dir) ──▶ <package_dir>/ swapped in whole:
        CSVs (1 decimal, NA), JSON (unrounded), captions.md, draft.md (optional template, unknown field = error),
        analysis_manifest.json (inputs + hash of every other emitted file)
        Package and standard outputs are both staged before either is swapped in.

Paper figures and dashboard share the same files (design Decision 4). Output is
deterministic: same bundle → byte-identical files (fixed SVG hash salt, no dates).
Never reads the store; only renders what it is given.
"""

from __future__ import annotations

import base64
import csv
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field, fields, is_dataclass
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
from evident.corpus import file_sha256
from evident.domain import Certainty, GateResult, Strength
from evident.paper import EDITION_RULE, BiasRow, PaperPackage, Proportion
from evident.themes import ThemeRow

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
_EXTRACTION_PROMPT_HEADER = ("# Extraction prompt\n\nSent once per page; the few-shot block shows the first set in "
                             "few_shot_examples.json (each guideline's examples excluded its own).\n\n```\n")
_FENCE_END = "\n```\n"
_JUDGE_PROMPT_HEADER = "# Duplicate judge prompt\n\nSent once per screened pair to each judge.\n\n```\n"
_HELDOUT_NOTE = "n={n} guidelines: imprecise; reported separately, not part of the development gate"
_OPTIONAL_NESTED = {(BiasRow, "correct"): Proportion}  # Optional dataclass fields, for stable CSV columns
_DEVELOPMENT_NOTE = ("the labelled guidelines also informed extractor development and tuning: "
                     "in-sample performance, not an external test")

# One colour per society everywhere (figures and dashboard), colour-blind safe (Okabe-Ito); ALL in neutral grey
_SOCIETY_COLOURS = {"ACP": "#0072B2", "ESICM": "#E69F00", "SCCM": "#009E73", _ALL: "#555555"}
_FALLBACK_COLOUR = "#9467bd"
_CERTAINTY_COLOURS = {Certainty.HIGH: "#08519c", Certainty.MODERATE: "#4292c6",
                      Certainty.LOW: "#9ecae1", Certainty.VERY_LOW: "#deebf7"}
_CERTAINTY_LABELS = {Certainty.HIGH: "High", Certainty.MODERATE: "Moderate",
                     Certainty.LOW: "Low", Certainty.VERY_LOW: "Very low"}
_STRENGTH_LABELS = {Strength.STRONG: "Strong", Strength.WEAK: "Weak / conditional"}
_ALL_GAP = 0.5              # extra axis units before ALL (figures 1–2)
_SPACER_RATIO = 0.15        # empty grid column before the ALL panel (figure 3)
_COLOURBAR_RATIO = 0.08
_WHITE_TEXT_SHARE = 50      # % of panel above which cell text turns white
_LABEL_HEADROOM = 15        # y units above the highest bar or whisker for its two-line label
_MARKER_AREA_PER_REC = 0.5  # pt² per graded recommendation (figure 4)
_SIZE_LEGEND = (10, 50, 250)
_SOCIETY_ALPHA = 0.6
_THEME_FIGURE = "fig5_themes"
_WIDE_FIGURES = ("fig3_strength_by_certainty", "fig4_trends", _THEME_FIGURE)  # multi-panel: full dashboard row
_THEME_ROW_INCHES = 0.3     # per theme row (figure 5)
_THEME_PANEL_INCHES = 0.9   # per specialty panel: title and certainty labels


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
    themes: list[ThemeRow] = field(default_factory=list)  # exploratory; empty → no theme figure or table
    search_cutoff: Optional[str] = None  # live map: last date searched, shown with the edition rule


class DraftFieldError(ValueError):
    """The draft template names a field the package does not export; no number is typed by hand."""


@dataclass(frozen=True)
class _Figure:
    id: str
    title: str
    caption: str


def render(bundle: PublishBundle, out_dir: str) -> list[str]:
    """Write every output; returns the final paths. A failure leaves the previous outputs intact."""
    staged = stage(bundle, out_dir)
    staged.install()
    return staged.paths


class StagedOutputs:
    """Rendered outputs not yet in place: checksum them, then install() or discard()."""

    def __init__(self, staging: str, out_dir: str, paths: list[str]):
        self._staging = staging
        self._out_dir = out_dir
        self.paths = paths  # final paths, e.g. <out_dir>/tables/headline.csv

    def checksums(self) -> dict[str, str]:
        """{path relative to out_dir: SHA-256} of the staged files, i.e. of the outputs once installed."""
        relative = [os.path.relpath(p, self._out_dir) for p in self.paths]
        return {rel: file_sha256(os.path.join(self._staging, rel)) for rel in relative}

    def install(self) -> None:
        _swap_in(self._staging, self._out_dir)

    def discard(self) -> None:
        """Drop the staging dir; a no-op after install()."""
        shutil.rmtree(self._staging, ignore_errors=True)


def stage(bundle: PublishBundle, out_dir: str) -> StagedOutputs:
    """Write every output to a staging dir; nothing in out_dir changes until install()."""
    staging, written = _stage_outputs(bundle, out_dir)
    return StagedOutputs(staging, out_dir, written)


def _stage_outputs(bundle: PublishBundle, out_dir: str) -> tuple[str, list[str]]:
    """Write every output to a staging dir in out_dir; returns it and the final paths. Nothing is swapped."""
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
    return staging, written


# ── tables ──────────────────────────────────────────────────────

def _write_tables(bundle: PublishBundle, directory: str) -> None:
    os.makedirs(directory)
    _write_csv(bundle.headline, os.path.join(directory, "headline.csv"))
    _write_csv(bundle.certainty, os.path.join(directory, "certainty.csv"))
    _write_csv(bundle.crosstab, os.path.join(directory, "strength_by_certainty.csv"))
    _write_csv(bundle.trends, os.path.join(directory, "trends.csv"))
    _write_csv(bundle.guidelines, os.path.join(directory, "guidelines.csv"))
    _write_dicts([_rec_dict(r) for r in bundle.recs], os.path.join(directory, "recommendations.csv"))
    if bundle.themes:
        _write_csv(bundle.themes, os.path.join(directory, "themes.csv"))


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
    """Short, distinct captions; the common corpus note and the provenance are stated once, beside them."""
    c = bundle.captions
    trend_note = (f"All editions: {c.n_recs_all_editions} recommendations, including "
                  f"{c.n_superseded_guidelines} superseded edition(s). Few guidelines per society and year; "
                  "marker area is proportional to n graded; the last year is incomplete.")
    whiskers = " Whiskers: 95% guideline-cluster bootstrap interval." if _strong_cis(bundle) else ""
    return [
        _Figure("fig1_certainty_by_society", "Certainty of evidence by society",
                "Share of current graded recommendations at each certainty level; n = certainty mapped."),
        _Figure("fig2_strong_by_society", "Strong recommendations by society",
                "Share of current graded recommendations that are strong (for or against); "
                f"n = strength mapped.{whiskers}"),
        _Figure("fig3_strength_by_certainty", "Strength by certainty",
                "Counts of current graded recommendations by strength and certainty (both mapped); "
                "shading = % of the panel."),
        _Figure("fig4_trends", "Trends by publication year",
                f"Share of strong and of high/moderate-certainty recommendations per first-online year. "
                f"{trend_note}"),
    ]


def _theme_figures(bundle: PublishBundle) -> list[_Figure]:
    """Exploratory theme figure, dashboard and figures/ only: not prespecified, so not in the paper package."""
    if not bundle.themes:
        return []
    return [_Figure(_THEME_FIGURE, "Certainty of evidence by theme (exploratory)",
                    "Themes from a reviewed taxonomy: drafted by an LLM from the recommendations' topics, edited "
                    "by a clinician. An LLM assigns each critical care recommendation from its text; an ACP "
                    "guideline covers one condition, its recommendations take that theme. Shading and text = % "
                    "of the theme's graded recommendations at each certainty level; n = certainty mapped. "
                    "Ordered by mean certainty, highest first; Other last.")]


def _common_caption(bundle: PublishBundle) -> str:
    """What a reader needs to read every figure: corpus, denominators, exclusions, joint count, caveats."""
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
        parts.append(f"{len(c.joint)} joint guideline(s) counted once, under one society (see provenance).")
    parts += _unvalidated_notes(bundle)
    return " ".join(parts)


def _provenance_note(bundle: PublishBundle) -> str:
    """Joint guidelines, held-out evaluation, snapshot, versions and validation: stated once, apart from the notes."""
    parts = [_joint_note(bundle.captions)] + _heldout_notes(bundle) + [_provenance(bundle)]
    return " ".join(p for p in parts if p)


def _joint_note(c: CaptionFacts) -> str:
    if not c.joint:
        return ""
    joint = "; ".join(f"{doi} ({society} with {_join(others)})" for doi, society, others in c.joint)
    return f"Joint guidelines, counted under the society listed: {joint}."


def _unvalidated_notes(bundle: PublishBundle) -> list[str]:
    """Societies without development labels and not evaluated on the held-out set: a caveat for every figure."""
    heldout = bundle.paper.heldout_societies if bundle.paper else ()
    return [f"{s}: extraction accuracy unvalidated (no labelled {s} guideline)."
            for s in bundle.captions.unvalidated_societies if s not in heldout]


def _heldout_notes(bundle: PublishBundle) -> list[str]:
    """Societies without development labels evaluated on the held-out set instead (package only)."""
    heldout = bundle.paper.heldout_societies if bundle.paper else ()
    n_dev = bundle.validation.n_guidelines if bundle.validation else 0
    return [f"{s}: evaluated separately on {bundle.paper.validation_heldout.n_guidelines} held-out "
            f"guidelines; not part of the {n_dev}-guideline development gate."
            for s in bundle.captions.unvalidated_societies if s in heldout]


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
              f"**Notes (all figures and Table 1).** {_common_caption(bundle)}", "",
              f"**Provenance.** {_provenance_note(bundle)}", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── figures ─────────────────────────────────────────────────────

def _write_figures(bundle: PublishBundle, directory: str) -> list[_Figure]:
    os.makedirs(directory)
    figures = _figures(bundle) + _theme_figures(bundle)
    painters = {
        "fig1_certainty_by_society": _paint_certainty,
        "fig2_strong_by_society": _paint_strong,
        "fig3_strength_by_certainty": _paint_crosstab,
        "fig4_trends": _paint_trends,
        _THEME_FIGURE: _paint_themes,
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
    ys = _positions(societies)
    figure = Figure(figsize=(6.5, 0.6 * len(societies) + 1.2))
    ax = figure.add_subplot()
    for level in _CERTAINTY_COLOURS:
        lefts = [_left_of(bundle.certainty, s, level) for s in societies]
        widths = [_pct_or_zero(_find(bundle.certainty, s, level).pct) for s in societies]
        ax.barh(ys, widths, left=lefts, color=_CERTAINTY_COLOURS[level], label=_CERTAINTY_LABELS[level],
                edgecolor="white")

    # n beside each bar, outside the 0–100 frame
    for y, s in zip(ys, societies):
        n = sum(r.n for r in bundle.certainty if r.society == s)
        ax.text(101, y, f"n={n}", va="center", clip_on=False)

    ax.set_yticks(ys, societies)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of graded recommendations")
    ax.invert_yaxis()
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    _drop_top_right(ax)
    figure.tight_layout()
    return figure


def _paint_strong(bundle: PublishBundle) -> Figure:
    """Bars of % strong per society + ALL; guideline-cluster CI whiskers when the package supplies them."""
    rows = bundle.headline
    names = [r.society for r in rows]
    xs = _positions(names)
    values = [_pct_or_zero(r.pct_strong) for r in rows]
    cis = _strong_cis(bundle)
    figure = Figure(figsize=(5, 3.2))
    ax = figure.add_subplot()
    ax.bar(xs, values, color=[_colour(n) for n in names])

    tops = []
    for x, r, value in zip(xs, rows, values):
        ci = cis.get(r.society)
        if ci:
            # A percentile interval need not contain the rounded estimate: clip, never a negative arm
            arms = [[max(0.0, value - ci[0])], [max(0.0, ci[1] - value)]]
            ax.errorbar(x, value, yerr=arms, color="black", capsize=3, linewidth=1)
        top = ci[1] if ci else value
        tops.append(top)
        label = "n/a" if r.pct_strong is None else f"{r.pct_strong:.1f}%"
        ax.text(x, top + 1.5, f"{label}\nn={r.n_graded - r.n_strength_unmapped}", ha="center", va="bottom")

    ax.set_xticks(xs, names)
    ax.set_ylim(0, max(tops, default=0) + _LABEL_HEADROOM)
    ax.set_ylabel("% strong")
    _drop_top_right(ax)
    figure.tight_layout()
    return figure


def _paint_crosstab(bundle: PublishBundle) -> Figure:
    """One heatmap panel per society + ALL: rows = strength, columns = certainty.

    Cell text = count; shading = % of the panel's recommendations on one 0–100 scale, comparable across panels.
    """
    societies = _societies(bundle.crosstab)
    levels = list(_CERTAINTY_COLOURS)
    strengths = list(_STRENGTH_LABELS)

    # Grid columns: panels, a narrow spacer before ALL, then the colourbar
    columns = []
    for i, s in enumerate(societies):
        if s == _ALL and i > 0:
            columns.append(_SPACER_RATIO)
        columns.append(1)
    columns.append(_COLOURBAR_RATIO)
    figure = Figure(figsize=(2.3 * len(societies) + 1.2, 2.4))
    grid = figure.add_gridspec(1, len(columns), width_ratios=columns)
    panel_columns = [i for i, w in enumerate(columns[:-1]) if w == 1]

    image = None
    for k, (column, society) in enumerate(zip(panel_columns, societies)):
        ax = figure.add_subplot(grid[0, column])
        counts = [[_find_cross(bundle.crosstab, society, st, lv).n for lv in levels] for st in strengths]
        total = sum(sum(row) for row in counts) or 1
        shares = [[100 * n / total for n in row] for row in counts]
        image = ax.imshow(shares, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for i, row in enumerate(counts):
            for j, n in enumerate(row):
                ax.text(j, i, str(n), ha="center", va="center",
                        color="white" if shares[i][j] > _WHITE_TEXT_SHARE else "black")
        ax.set_title(society)
        ax.set_xticks(range(len(levels)), [_CERTAINTY_LABELS[lv] for lv in levels], rotation=45, ha="right")
        ax.set_yticks(range(len(strengths)), [_STRENGTH_LABELS[st] if k == 0 else "" for st in strengths])

    if image is not None:
        figure.colorbar(image, cax=figure.add_subplot(grid[0, -1]), label="% of panel")
    figure.tight_layout()
    return figure


def _paint_trends(bundle: PublishBundle) -> Figure:
    """Two panels (% strong, % high/moderate) by year; one line per society, ALL dashed and emphasised.

    Marker area is proportional to n graded, so sparse years (n of 2–4) draw the eye less than full ones.
    """
    figure = Figure(figsize=(7.5, 3.2))
    left, right = figure.subplots(1, 2, sharex=True)
    for ax, attr, title in ((left, "pct_strong", "% strong"), (right, "pct_high_moderate", "% high/moderate")):
        for society in _societies(bundle.trends):
            rows = [r for r in bundle.trends if r.society == society]
            years = [r.year for r in rows]
            values = [math.nan if getattr(r, attr) is None else getattr(r, attr) for r in rows]
            is_all = society == _ALL
            alpha = 1.0 if is_all else _SOCIETY_ALPHA
            ax.plot(years, values, label=society, color=_colour(society), linestyle="--" if is_all else "-",
                    linewidth=2 if is_all else 1, alpha=alpha, zorder=3 if is_all else 2)
            ax.scatter(years, values, s=[_marker_area(r.n_graded) for r in rows], color=_colour(society),
                       alpha=alpha, zorder=3 if is_all else 2, linewidths=0)
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.xaxis.get_major_locator().set_params(integer=True)
        _drop_top_right(ax)

    if bundle.trends:
        left.legend(frameon=False, fontsize=7)
        sizes = [right.scatter([], [], s=_marker_area(n), color="grey", linewidths=0) for n in _SIZE_LEGEND]
        right.legend(sizes, [str(n) for n in _SIZE_LEGEND], title="n graded", frameon=False, fontsize=7,
                     title_fontsize=7, labelspacing=1.2)
    figure.tight_layout()
    return figure


def _paint_themes(bundle: PublishBundle) -> Figure:
    """One heatmap per specialty, stacked: rows = themes, columns = certainty; shading and text = row %."""
    specialties = []
    for r in bundle.themes:
        if r.specialty not in specialties:
            specialties.append(r.specialty)
    levels = list(_CERTAINTY_COLOURS)
    n_themes = [len({r.theme for r in bundle.themes if r.specialty == s}) for s in specialties]

    figure = Figure(figsize=(7.5, _THEME_ROW_INCHES * sum(n_themes) + _THEME_PANEL_INCHES * len(specialties)))
    grid = figure.add_gridspec(len(specialties), 2, height_ratios=n_themes, width_ratios=[1, _COLOURBAR_RATIO])
    image = None
    for i, s in enumerate(specialties):
        rows = [r for r in bundle.themes if r.specialty == s]
        themes = sorted({r.theme for r in rows})
        pcts = [[_find_theme(rows, t, lv).pct for lv in levels] for t in themes]
        shares = [[_pct_or_zero(p) for p in row] for row in pcts]

        ax = figure.add_subplot(grid[i, 0])
        image = ax.imshow(shares, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for y, row in enumerate(pcts):
            for x, pct in enumerate(row):
                ax.text(x, y, "–" if pct is None else f"{pct:.0f}", ha="center", va="center",
                        color="white" if _pct_or_zero(pct) > _WHITE_TEXT_SHARE else "black")

        # Row label: theme terms and the certainty-mapped denominator, e.g. "sepsis · cultures (n=84)"
        first = [_find_theme(rows, t, levels[0]) for t in themes]
        mapped = [sum(r.n for r in rows if r.theme == t) for t in themes]
        ax.set_yticks(range(len(themes)), [f"{f.label} (n={n})" for f, n in zip(first, mapped)])
        ax.set_xticks(range(len(levels)), [_CERTAINTY_LABELS[lv] for lv in levels])
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        ax.set_title(f"{s} ({_join(rows[0].societies)})", loc="left", pad=18)

    if image is not None:
        figure.colorbar(image, cax=figure.add_subplot(grid[:, 1]), label="% of theme")
    figure.tight_layout()
    return figure


def _positions(societies: list[str]) -> list[float]:
    """Axis positions with a gap before ALL, so the pooled row does not read as a fourth society."""
    return [i + (_ALL_GAP if s == _ALL else 0) for i, s in enumerate(societies)]


def _strong_cis(bundle: PublishBundle) -> dict[str, tuple[float, float]]:
    """% strong guideline-cluster CI per society, from Table 1; empty without the paper package."""
    if bundle.paper is None:
        return {}
    return {r.society: (r.strong.ci_low, r.strong.ci_high) for r in bundle.paper.table1
            if r.strong.ci_low is not None and r.strong.ci_high is not None}


def _marker_area(n_graded: int) -> float:
    return _MARKER_AREA_PER_REC * n_graded


def _drop_top_right(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


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


def _find_theme(rows: list[ThemeRow], theme: int, level: Certainty) -> ThemeRow:
    return next(r for r in rows if r.theme == theme and r.certainty == level)


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
        heldout=_heldout_view(bundle),
        facts=c,
        headline=[{k: _cell(v) for k, v in asdict(r).items()} for r in bundle.headline],
        figures=[{"title": f.title, "caption": f.caption, "wide": f.id in _WIDE_FIGURES,
                  "src": _svg_data_uri(os.path.join(figure_dir, f.id + ".svg"))}
                 for f in figures],
        common_caption=_common_caption(bundle),
        joint_note=_joint_note(c),  # the rest of the provenance is in the validation table and line beside it
        search_cutoff=bundle.search_cutoff,
        edition_rule=EDITION_RULE,
        all_label=_ALL,
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


def _heldout_view(bundle: PublishBundle) -> Optional[dict]:
    """Held-out validation row (package only): micro P/R/F1 and grade accuracy, Wilson intervals."""
    if bundle.paper is None:
        return None
    h = bundle.paper.validation_heldout
    return {"n": h.n_guidelines, "societies": _join(bundle.paper.heldout_societies),
            "precision": _num(h.precision), "precision_ci": _ci(h.precision_ci),
            "recall": _num(h.recall), "recall_ci": _ci(h.recall_ci), "f1": _num(h.f1),
            "combined": _num(h.combined_accuracy), "combined_ci": _ci(h.combined_ci), "note": h.interval_note}


def _svg_data_uri(path: str) -> str:
    """An <img> data URI, not inline SVG: matplotlib ids would collide across figures in one page."""
    with open(path, "rb") as f:
        return "data:image/svg+xml;base64," + base64.b64encode(f.read()).decode("ascii")


def _embedded_json(data: dict) -> str:
    """JSON safe inside <script>: every '<' escaped, so '</script>' in a text can't close the tag."""
    return json.dumps(data, sort_keys=True, ensure_ascii=False).replace("<", "\\u003c")


# ── analysis package ────────────────────────────────────────────

def render_package(bundle: PublishBundle, package_dir: str, out_dir: str) -> tuple[list[str], list[str]]:
    """Write the paper package and the standard outputs; returns both final path lists.

    Both are staged before either is swapped in, so a failure in either leaves both previous versions intact.
    """
    if bundle.paper is None:
        raise ValueError("render_package needs a bundle with a paper payload")
    problem = package_target_problem(package_dir)
    if problem:
        raise ValueError(problem)

    parent = os.path.dirname(os.path.abspath(package_dir))
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=_PACKAGE_STAGING_PREFIX, dir=parent)
    try:
        _write_package(bundle, staging)
        outputs_staging, written = _stage_outputs(bundle, out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    names = sorted(os.listdir(staging))
    _swap_dir(staging, os.path.abspath(package_dir))
    _swap_in(outputs_staging, out_dir)
    return [os.path.join(package_dir, n) for n in names], written


def package_target_problem(package_dir: str) -> Optional[str]:
    """Why package_dir cannot be replaced, or None: it must be missing, empty or a previous package.

    The swap deletes the old directory, so e.g. `--package out` (the live DB beside the outputs) is refused.
    """
    if not os.path.exists(package_dir):
        return None
    if not os.path.isdir(package_dir):
        return f"package target {package_dir} is not a directory"
    names = os.listdir(package_dir)
    if names and _MANIFEST_FILE not in names:
        return (f"package target {package_dir} is not empty and not a previous package (no {_MANIFEST_FILE}); "
                "refusing to replace it")
    return None


def _write_package(bundle: PublishBundle, directory: str) -> None:
    p = bundle.paper
    _write_package_csv(p.table1, os.path.join(directory, "table1_society.csv"))
    _write_package_csv(p.table2, os.path.join(directory, "table2_certainty_strength.csv"))
    _write_package_csv(p.guideline_mean, os.path.join(directory, "table_s1_guideline_mean.csv"))
    _write_package_csv(p.table1_without_dedup, os.path.join(directory, "table_s2_without_dedup.csv"))
    _write_package_csv(p.figure_data, os.path.join(directory, "figure_data.csv"))
    _write_package_csv(p.corpus_flow, os.path.join(directory, "corpus_flow.csv"))
    _write_package_csv(p.excluded, os.path.join(directory, "excluded_guidelines.csv"))
    _write_package_csv(p.raw_mapping, os.path.join(directory, "raw_harmonized_mapping.csv"))
    _write_package_csv(p.spotcheck_bias, os.path.join(directory, "spotcheck_bias.csv"))
    _write_package_csv(p.duplicates, os.path.join(directory, "duplicates.csv"))

    # Supplement: model specification (eAppendix 5) and the duplicate judges' prompt (eAppendix 2)
    _write_text(_EXTRACTION_PROMPT_HEADER + p.extraction_prompt + _FENCE_END, os.path.join(directory, "extraction_prompt.md"))
    _write_json(p.few_shot_sets, os.path.join(directory, "few_shot_examples.json"))
    _write_text(_JUDGE_PROMPT_HEADER + p.judge_prompt + _FENCE_END, os.path.join(directory, "dedup_judge_prompt.md"))

    exports = _package_json(bundle)
    for name, data in exports.items():
        _write_json(data, os.path.join(directory, name + ".json"))

    _write_package_captions(bundle, os.path.join(directory, "captions.md"))

    # The manuscript is optional: a clean checkout has no private template
    if p.draft_template is not None:
        with open(os.path.join(directory, _DRAFT_FILE), "w", encoding="utf-8") as f:
            f.write(_render_draft(bundle, exports))

    # Last: hashes of every other file; the manifest never lists itself
    outputs = {name: file_sha256(os.path.join(directory, name)) for name in sorted(os.listdir(directory))}
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
                 list(flat[0]) if flat else [])


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
    template_dir, name = os.path.split(os.path.abspath(p.draft_template))
    env = Environment(loader=FileSystemLoader(template_dir), undefined=StrictUndefined, keep_trailing_newline=True,
                      autoescape=False)
    env.filters.update(pct=_fmt_pct, ci=_fmt_ci_pct, p3=_fmt_p3, ci3=_fmt_ci3)
    try:
        return env.get_template(name).render(**_draft_context(bundle, exports))
    except UndefinedError as e:
        raise DraftFieldError(f"{p.draft_template}: {e}") from e


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
        "table_s2_without_dedup": {r.society: _flat(r) for r in p.table1_without_dedup},
        "few_shot_examples": p.few_shot_sets,
        "corpus_flow": flow,
        "excluded_guidelines": [_flat(r) for r in p.excluded],
        "raw_harmonized_mapping": [_flat(r) for r in p.raw_mapping],
        "spotcheck_bias": {r.metric.value: _flat(r) for r in p.spotcheck_bias},
        "captions": {f.id: f.caption for f in _figures(bundle)},
        "notes": {"common": _common_caption(bundle), "provenance": _provenance_note(bundle)},
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
        "**Table S2. Without duplicate removal.** Table 1 recomputed with every within-guideline copy kept; "
        "if every merge were wrong, Table 1 would equal this table.", "",
        f"**Notes.** {_common_caption(bundle)}", "",
        f"**Provenance.** {_provenance_note(bundle)}", "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_text(text: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


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
    """Replace each output subdir by its staged version, then drop the staging dir.

    All or nothing: a failed rename puts back every subdir already set aside, e.g. new tables
    beside old figures is never left behind.
    """
    set_aside = []
    try:
        for sub in _SUBDIRS:
            final = os.path.join(out_dir, sub)
            _set_aside(final)
            set_aside.append(final)
            os.rename(os.path.join(staging, sub), final)
    except BaseException:
        for final in reversed(set_aside):
            _put_back(final)
        shutil.rmtree(staging, ignore_errors=True)
        raise

    for final in set_aside:
        shutil.rmtree(final + _OLD_SUFFIX, ignore_errors=True)
    os.rmdir(staging)


def _set_aside(final: str) -> None:
    """Move the previous <final> to <final>.old (if there is one)."""
    old = final + _OLD_SUFFIX
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(final):
        os.rename(final, old)


def _put_back(final: str) -> None:
    """Undo _set_aside: drop any new <final>, restore <final>.old."""
    old = final + _OLD_SUFFIX
    shutil.rmtree(final, ignore_errors=True)
    if os.path.exists(old):
        os.rename(old, final)


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
