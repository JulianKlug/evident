#!/usr/bin/env python3
"""Generate a PDF performance report for the best extraction config.

PRELIMINARY: exploratory report over the March 2026 A/B benchmark CSVs.
Guideline metadata is hard-coded below. Superseded by the publishing
layer (roadmap M3, docs/designs/living-evidence-map.md); do not extend.
"""

from __future__ import annotations

import csv
import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


DATA_DIR = "artifacts/ab_tests"
OUTPUT = "artifacts/performance_report.pdf"

# Configs to include in comparison (order matters)
CONFIGS = OrderedDict([
    ("sc_adaptive_ml_filter", "SC + ML Filter"),
    ("sc_adaptive", "SC Adaptive"),
    ("baseline", "Baseline (few-shot)"),
])

# Friendly guideline names
GUIDELINE_NAMES = {
    "WND8NBNA": "ACP: WND8NBNA",
    "XLMXNL32": "ACP: XLMXNL32",
    "8J2P9MD8": "ACP: 8J2P9MD8",
    "8V9WED94": "ACP: 8V9WED94",
    "XK8ZAXYM": "ACP: XK8ZAXYM",
    "48AJE2AR": "ACP: 48AJE2AR",
    "XBAJSPZE": "ACP: XBAJSPZE",
    "PAEHSPH3": "ACP: PAEHSPH3",
    "89499SID": "ACP: 89499SID",
    "CMCZFLU4": "ERS: CMCZFLU4",
    "BDYDTUHA": "ERS: BDYDTUHA",
    "10_1007_s00134-024-07369-9": "ICU: 07369-9",
    "10_1007_s00134-025-07840-1": "ICU: 07840-1",
    "10_1007_s00134-025-08058-x": "ICU: 08058-x",
}

SCHEME_MAP = {
    "WND8NBNA": "GRADE", "XLMXNL32": "GRADE", "8J2P9MD8": "GRADE",
    "8V9WED94": "GRADE", "XK8ZAXYM": "GRADE", "48AJE2AR": "GRADE",
    "XBAJSPZE": "GRADE", "PAEHSPH3": "GRADE", "89499SID": "GRADE",
    "CMCZFLU4": "GRADE", "BDYDTUHA": "ABCD_123",
    "10_1007_s00134-024-07369-9": "GRADE",
    "10_1007_s00134-025-07840-1": "GRADE",
    "10_1007_s00134-025-08058-x": "GRADE",
}

SOURCE_MAP = {
    "WND8NBNA": "ACP", "XLMXNL32": "ACP", "8J2P9MD8": "ACP",
    "8V9WED94": "ACP", "XK8ZAXYM": "ACP", "48AJE2AR": "ACP",
    "XBAJSPZE": "ACP", "PAEHSPH3": "ACP", "89499SID": "ACP",
    "CMCZFLU4": "ERS", "BDYDTUHA": "ERS",
    "10_1007_s00134-024-07369-9": "ICU",
    "10_1007_s00134-025-07840-1": "ICU",
    "10_1007_s00134-025-08058-x": "ICU",
}


def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["guideline_key"] == "NI9RV3E7":
                continue  # exclude vision-only guideline
            rows.append(row)
    return rows


def fmt(val, decimals=3):
    return f"{float(val):.{decimals}f}"


def add_title_page(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.5, 0.6, "Clinical Guideline Recommendation Extraction",
            ha="center", va="center", fontsize=22, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.50, "Performance Report",
            ha="center", va="center", fontsize=18, color="#444444",
            transform=ax.transAxes)
    ax.text(0.5, 0.38, "Best Configuration: Self-Consistency + ML Filter",
            ha="center", va="center", fontsize=14, color="#666666",
            transform=ax.transAxes)
    ax.text(0.5, 0.30, "Model: qwen3:14b  |  14 Text Guidelines  |  Avg F1 = 0.860",
            ha="center", va="center", fontsize=12, color="#888888",
            transform=ax.transAxes)
    pdf.savefig(fig)
    plt.close(fig)


def add_overall_table(pdf, data_by_config):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.95, "Overall Performance Comparison",
            ha="center", va="top", fontsize=16, fontweight="bold",
            transform=ax.transAxes)

    headers = ["Configuration", "Avg F1", "Avg P", "Avg R", "Grade", "Level", "Combined"]
    rows_data = []
    for config_key, config_name in CONFIGS.items():
        rows = data_by_config[config_key]
        f1s = [float(r["f1"]) for r in rows]
        ps = [float(r["precision"]) for r in rows]
        rs = [float(r["recall"]) for r in rows]
        gs = [float(r["grade_accuracy"]) for r in rows]
        ls = [float(r["level_accuracy"]) for r in rows]
        cs = [float(r["combined_accuracy"]) for r in rows]
        rows_data.append([
            config_name,
            fmt(np.mean(f1s)),
            fmt(np.mean(ps)),
            fmt(np.mean(rs)),
            fmt(np.mean(gs)),
            fmt(np.mean(ls)),
            fmt(np.mean(cs)),
        ])

    col_widths = [0.19, 0.10, 0.10, 0.10, 0.11, 0.11, 0.13]
    table = ax.table(
        cellText=rows_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 2.0)

    # Style header
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Highlight best row
    for j in range(len(headers)):
        table[1, j].set_facecolor("#d5f5e3")

    pdf.savefig(fig)
    plt.close(fig)


def add_per_guideline_table(pdf, best_rows):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.97, "Per-Guideline Performance (SC + ML Filter)",
            ha="center", va="top", fontsize=16, fontweight="bold",
            transform=ax.transAxes)

    headers = ["Guideline", "Source", "Scheme", "N GT", "N Ext", "P", "R", "F1", "Grade", "Level", "Combined"]
    rows_data = []
    for r in best_rows:
        key = r["guideline_key"]
        rows_data.append([
            GUIDELINE_NAMES.get(key, key),
            SOURCE_MAP.get(key, "?"),
            SCHEME_MAP.get(key, "?"),
            r["n_gt"],
            r["n_extracted"],
            fmt(float(r["precision"])),
            fmt(float(r["recall"])),
            fmt(float(r["f1"])),
            fmt(float(r["grade_accuracy"])),
            fmt(float(r["level_accuracy"])),
            fmt(float(r["combined_accuracy"])),
        ])

    # Add average row
    f1s = [float(r["f1"]) for r in best_rows]
    ps = [float(r["precision"]) for r in best_rows]
    rs = [float(r["recall"]) for r in best_rows]
    gs = [float(r["grade_accuracy"]) for r in best_rows]
    ls = [float(r["level_accuracy"]) for r in best_rows]
    cs = [float(r["combined_accuracy"]) for r in best_rows]
    n_gt_total = sum(int(r["n_gt"]) for r in best_rows)
    n_ext_total = sum(int(r["n_extracted"]) for r in best_rows)
    rows_data.append([
        "AVERAGE", "", "", str(n_gt_total), str(n_ext_total),
        fmt(np.mean(ps)), fmt(np.mean(rs)), fmt(np.mean(f1s)),
        fmt(np.mean(gs)), fmt(np.mean(ls)), fmt(np.mean(cs)),
    ])

    col_widths = [0.13, 0.06, 0.08, 0.06, 0.06, 0.07, 0.07, 0.07, 0.08, 0.08, 0.10]
    table = ax.table(
        cellText=rows_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)

    # Style header
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=8)

    # Color F1 cells by value
    f1_col = 7
    for i, r in enumerate(best_rows):
        f1 = float(r["f1"])
        if f1 >= 0.9:
            color = "#d5f5e3"
        elif f1 >= 0.7:
            color = "#fef9e7"
        else:
            color = "#fadbd8"
        table[i + 1, f1_col].set_facecolor(color)

    # Style average row
    avg_row = len(rows_data)
    for j in range(len(headers)):
        table[avg_row, j].set_facecolor("#eaf2f8")
        table[avg_row, j].set_text_props(fontweight="bold")

    pdf.savefig(fig)
    plt.close(fig)


def add_f1_bar_chart(pdf, data_by_config, best_rows):
    fig, ax = plt.subplots(figsize=(11, 8.5))

    guidelines = [r["guideline_key"] for r in best_rows]
    labels = [GUIDELINE_NAMES.get(g, g) for g in guidelines]
    x = np.arange(len(guidelines))
    width = 0.25

    colors = ["#27ae60", "#3498db", "#e74c3c"]
    for i, (config_key, config_name) in enumerate(CONFIGS.items()):
        rows = data_by_config[config_key]
        f1_map = {r["guideline_key"]: float(r["f1"]) for r in rows}
        f1_vals = [f1_map.get(g, 0) for g in guidelines]
        ax.bar(x + i * width, f1_vals, width, label=config_name, color=colors[i], alpha=0.85)

    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("F1 Score by Guideline — Configuration Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.860, color="#27ae60", linestyle="--", alpha=0.5, label="Best Avg F1 (0.860)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    pdf.savefig(fig)
    plt.close(fig)


def add_pr_scatter(pdf, best_rows):
    fig, ax = plt.subplots(figsize=(11, 8.5))

    # Collect points for label placement
    points = []
    for r in best_rows:
        p = float(r["precision"])
        rec = float(r["recall"])
        key = r["guideline_key"]
        source = SOURCE_MAP.get(key, "?")
        color = {"ACP": "#3498db", "ERS": "#e67e22", "ICU": "#9b59b6"}[source]
        marker = {"ACP": "o", "ERS": "s", "ICU": "D"}[source]
        ax.scatter(rec, p, c=color, marker=marker, s=120, edgecolors="white", linewidth=0.5, zorder=3)
        points.append((rec, p, GUIDELINE_NAMES.get(key, key).split(": ")[1]))

    # Simple greedy label placement to avoid overlaps
    placed = []
    for rec, p, label in points:
        # Try different offsets
        offsets = [(6, 6), (-6, 8), (6, -10), (-40, -10), (-40, 8), (6, 14), (-40, -18)]
        best_offset = offsets[0]
        for ox, oy in offsets:
            # Check rough overlap with already-placed labels
            candidate = (rec + ox * 0.003, p + oy * 0.003)
            overlap = False
            for px, py in placed:
                if abs(candidate[0] - px) < 0.06 and abs(candidate[1] - py) < 0.03:
                    overlap = True
                    break
            if not overlap:
                best_offset = (ox, oy)
                placed.append(candidate)
                break
        else:
            placed.append((rec + best_offset[0] * 0.003, p + best_offset[1] * 0.003))

        ax.annotate(label, (rec, p), fontsize=7, ha="left", va="bottom",
                    xytext=best_offset, textcoords="offset points")

    # F1 iso-lines
    for f1_val in [0.6, 0.7, 0.8, 0.9, 1.0]:
        r_range = np.linspace(0.01, 1.0, 200)
        p_range = (f1_val * r_range) / (2 * r_range - f1_val)
        mask = (p_range > 0) & (p_range <= 1.05)
        ax.plot(r_range[mask], p_range[mask], "--", color="gray", alpha=0.3, linewidth=0.8)
        # Label at right edge
        idx = np.where(mask)[0]
        if len(idx) > 0:
            li = idx[-1]
            if p_range[li] <= 1.02:
                ax.text(r_range[li], p_range[li], f"F1={f1_val}", fontsize=7, color="gray", alpha=0.5)

    # Legend
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    legend_elements = [
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=10, label="ACP"),
        mlines.Line2D([0], [0], marker="s", color="w", markerfacecolor="#e67e22", markersize=10, label="ERS"),
        mlines.Line2D([0], [0], marker="D", color="w", markerfacecolor="#9b59b6", markersize=10, label="ICU"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision vs Recall by Guideline (SC + ML Filter)", fontsize=14, fontweight="bold")
    ax.set_xlim(0.35, 1.08)
    ax.set_ylim(0.55, 1.08)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    pdf.savefig(fig)
    plt.close(fig)


def add_source_breakdown(pdf, best_rows):
    fig, axes = plt.subplots(1, 3, figsize=(11, 8.5))
    fig.suptitle("Performance by Guideline Source (SC + ML Filter)",
                 fontsize=14, fontweight="bold", y=0.95)

    sources = ["ACP", "ERS", "ICU"]
    metrics = ["f1", "precision", "recall", "grade_accuracy", "level_accuracy", "combined_accuracy"]
    metric_labels = ["F1", "Precision", "Recall", "Grade", "Level", "Combined"]
    colors = ["#27ae60", "#3498db", "#e67e22", "#9b59b6", "#e74c3c", "#1abc9c"]

    for idx, source in enumerate(sources):
        ax = axes[idx]
        src_rows = [r for r in best_rows if SOURCE_MAP.get(r["guideline_key"]) == source]
        vals = []
        for m in metrics:
            vals.append(np.mean([float(r[m]) for r in src_rows]))

        bars = ax.bar(metric_labels, vals, color=colors, alpha=0.85)
        ax.set_title(f"{source} ({len(src_rows)} guidelines)", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig)
    plt.close(fig)


def main():
    # Load data
    data_by_config = {}
    for config_key in CONFIGS:
        path = os.path.join(DATA_DIR, f"{config_key}.csv")
        data_by_config[config_key] = load_csv(path)

    best_rows = data_by_config["sc_adaptive_ml_filter"]

    os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)

    with PdfPages(OUTPUT) as pdf:
        add_title_page(pdf)
        add_overall_table(pdf, data_by_config)
        add_per_guideline_table(pdf, best_rows)
        add_f1_bar_chart(pdf, data_by_config, best_rows)
        add_pr_scatter(pdf, best_rows)
        add_source_breakdown(pdf, best_rows)

    print(f"Report saved to {OUTPUT}")


if __name__ == "__main__":
    main()
