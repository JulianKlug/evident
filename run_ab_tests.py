#!/usr/bin/env python3
"""A/B testing script for extraction pipeline features.

Runs each feature variant on all available datasets and compares against
the baseline (qwen3:14b few-shot + normalize).

Usage:
    python run_ab_tests.py                          # Run all tests
    python run_ab_tests.py --skip-low-priority      # Skip json, two_pass, ensemble
    python run_ab_tests.py --skip-ensemble          # Skip ensemble only
    python run_ab_tests.py --only baseline,verify   # Run only named tests
    python run_ab_tests.py --dry-run                # Print what would run
"""

import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

os.environ["PYTHONUNBUFFERED"] = "1"

import pandas as pd

from extraction.benchmark import (
    BenchmarkEntry, _BioLORDSimilarityModel, filter_available_datasets,
    _build_strategy,
)
from extraction.datasets import load_all_datasets
from extraction.extractor import extract_guideline
from extraction.llm_client import OllamaClient
from evaluation.evaluate import evaluate_extraction

OUTPUT_DIR = "artifacts/ab_tests"
NI9RV3E7_KEY = "NI9RV3E7"
DEFAULT_MODEL = "qwen3:14b"


@dataclass
class ABTestConfig:
    """Configuration for a single A/B test variant."""
    name: str
    description: str
    priority: int  # 0=baseline, 1-6=high, 7-9=low
    prior_conclusion: str  # what we expected from previous experiments
    # Extraction kwargs overrides (relative to baseline)
    strategies: Optional[list] = None  # None = use baseline default ["few_shot"]
    normalize: Optional[bool] = None  # None = use baseline default True
    pages_per_chunk: int = 1
    output_format: str = "pipe"
    auto_vision: bool = False
    vision_model: str = "mistral-small3.2:24b"
    self_consistency: bool = False
    n_samples: int = 3
    sc_temperature: float = 0.3
    consensus: int = 2
    ensemble: bool = False
    ensemble_models: Optional[list] = None
    two_pass: bool = False
    verify: bool = False
    verify_threshold: float = 0.5
    post_filter: bool = False
    filter_model: str = "qwen3:8b"
    adaptive_threshold: bool = False


# ── Test Registry ──────────────────────────────────────────────────────────

AB_TESTS = [
    ABTestConfig(
        name="baseline",
        description="few-shot + normalize (best config)",
        priority=0,
        prior_conclusion="—",
    ),
    ABTestConfig(
        name="no_normalize",
        description="Drop normalization",
        priority=1,
        prior_conclusion="normalization helps grade accuracy",
        normalize=False,
    ),
    ABTestConfig(
        name="zero_shot",
        description="Zero-shot instead of few-shot",
        priority=2,
        prior_conclusion="worse than few-shot",
        strategies=["zero_shot"],
    ),
    ABTestConfig(
        name="verify",
        description="Token overlap verification",
        priority=3,
        prior_conclusion="near-zero impact (+0.001 P)",
        verify=True,
    ),
    ABTestConfig(
        name="auto_vision",
        description="Auto-vision with mistral",
        priority=4,
        prior_conclusion="helps NI9RV3E7 (0.00→0.65), neutral elsewhere",
        auto_vision=True,
        vision_model="mistral-small3.2:24b",
    ),
    ABTestConfig(
        name="chunking_3page",
        description="3-page chunks",
        priority=5,
        prior_conclusion="hurt ACP precision",
        pages_per_chunk=3,
    ),
    ABTestConfig(
        name="self_consistency",
        description="3 samples, consensus 2/3",
        priority=6,
        prior_conclusion="avg F1 -0.05, hurts BDYDTUHA recall",
        self_consistency=True,
        n_samples=3,
        sc_temperature=0.3,
        consensus=2,
    ),
    ABTestConfig(
        name="json_output",
        description="JSON structured output",
        priority=7,
        prior_conclusion="avg F1=0.47, over-extracts badly",
        output_format="json",
    ),
    ABTestConfig(
        name="two_pass",
        description="Two-pass with llama3.2",
        priority=8,
        prior_conclusion="avg F1=0.47, classifier too conservative",
        two_pass=True,
    ),
    ABTestConfig(
        name="ensemble",
        description="qwen3:14b + deepseek-r1:32b",
        priority=9,
        prior_conclusion="avg F1=0.42, too many false positives",
        ensemble=True,
        ensemble_models=["qwen3:14b", "deepseek-r1:32b"],
    ),
    ABTestConfig(
        name="json_schema",
        description="JSON with Ollama schema enforcement (re-test)",
        priority=10,
        prior_conclusion="Prior F1=0.47, re-testing with schema enforcement",
        output_format="json",
    ),
    ABTestConfig(
        name="post_filter",
        description="Binary classification filter (qwen3:8b)",
        priority=11,
        prior_conclusion="New — expected to improve precision",
        post_filter=True,
        filter_model="qwen3:8b",
    ),
    ABTestConfig(
        name="sc_post_filter",
        description="Self-consistency + post-filter",
        priority=12,
        prior_conclusion="New — SC precision + filter precision",
        self_consistency=True,
        n_samples=3,
        sc_temperature=0.3,
        consensus=2,
        post_filter=True,
        filter_model="qwen3:8b",
    ),
    ABTestConfig(
        name="sc_adaptive",
        description="Self-consistency with adaptive threshold",
        priority=13,
        prior_conclusion="New — fix small/large dataset SC issues",
        self_consistency=True,
        n_samples=3,
        sc_temperature=0.3,
        adaptive_threshold=True,
    ),
    ABTestConfig(
        name="sc_adaptive_post_filter",
        description="Adaptive SC + post-filter",
        priority=14,
        prior_conclusion="New — adaptive SC + classification filter",
        self_consistency=True,
        n_samples=3,
        sc_temperature=0.3,
        adaptive_threshold=True,
        post_filter=True,
        filter_model="qwen3:8b",
    ),
]


def _get_csv_path(test_name):
    return os.path.join(OUTPUT_DIR, f"{test_name}.csv")


def _load_completed_keys(test_name):
    """Load already-completed guideline keys for resumability."""
    path = _get_csv_path(test_name)
    if not os.path.isfile(path):
        return set()
    try:
        df = pd.read_csv(path)
        return set(df["guideline_key"].unique())
    except Exception:
        return set()


def _append_entry(test_name, entry):
    """Append a single BenchmarkEntry to the test's CSV file."""
    path = _get_csv_path(test_name)
    row = vars(entry)
    row["test_name"] = test_name
    df = pd.DataFrame([row])
    header = not os.path.isfile(path)
    df.to_csv(path, mode="a", header=header, index=False)


def run_single_test(config, datasets, similarity_model):
    """Run one A/B test config on all datasets. Returns list of BenchmarkEntry."""
    completed = _load_completed_keys(config.name)
    remaining = [ds for ds in datasets if ds.key not in completed]

    if not remaining:
        print(f"\n[A/B: {config.name}] All {len(datasets)} guidelines already done, skipping.",
              flush=True)
        # Load existing results
        path = _get_csv_path(config.name)
        if os.path.isfile(path):
            df = pd.read_csv(path)
            return [BenchmarkEntry(**{k: row[k] for k in BenchmarkEntry.__dataclass_fields__})
                    for _, row in df.iterrows()]
        return []

    if completed:
        print(f"\n[A/B: {config.name}] Resuming: {len(completed)} done, {len(remaining)} remaining.",
              flush=True)

    strategies = config.strategies or ["few_shot"]
    normalize = config.normalize if config.normalize is not None else True
    strategy_name = strategies[0]

    client = OllamaClient(model=DEFAULT_MODEL)
    entries = []

    # Load any previously completed entries
    path = _get_csv_path(config.name)
    if os.path.isfile(path):
        try:
            df = pd.read_csv(path)
            entries = [BenchmarkEntry(**{k: row[k] for k in BenchmarkEntry.__dataclass_fields__})
                       for _, row in df.iterrows()]
        except Exception:
            pass

    for i, ds in enumerate(remaining):
        idx = len(completed) + i + 1
        total = len(datasets)

        print(f"[A/B: {config.name}] ({idx}/{total}) {ds.dataset_name}:{ds.key}",
              end="  ", flush=True)

        strategy = _build_strategy(strategy_name, ds.grading_scheme, ds.key)

        start = time.time()
        try:
            if config.self_consistency:
                from extraction.self_consistency import self_consistency_extract
                result = self_consistency_extract(
                    source=ds.doi,
                    strategy=strategy,
                    client=client,
                    similarity_model=similarity_model,
                    n_samples=config.n_samples,
                    temperature=config.sc_temperature,
                    consensus_threshold=config.consensus,
                    pages_per_chunk=config.pages_per_chunk,
                    output_format=config.output_format,
                    normalize=normalize,
                    adaptive_threshold=config.adaptive_threshold,
                    post_filter=config.post_filter,
                    filter_model=config.filter_model,
                )
            elif config.ensemble:
                from extraction.ensemble import ensemble_extract
                result = ensemble_extract(
                    source=ds.doi,
                    strategy=strategy,
                    models=config.ensemble_models,
                    pages_per_chunk=config.pages_per_chunk,
                    output_format=config.output_format,
                    normalize=normalize,
                )
            elif config.auto_vision:
                from extraction.vision_extractor import auto_vision_extract_guideline
                result = auto_vision_extract_guideline(
                    source=ds.doi,
                    strategy=strategy,
                    client=client,
                    vision_model=config.vision_model,
                    pages_per_chunk=config.pages_per_chunk,
                    output_format=config.output_format,
                    normalize=normalize,
                    verify=config.verify,
                    verify_threshold=config.verify_threshold,
                    post_filter=config.post_filter,
                    filter_model=config.filter_model,
                )
            elif config.two_pass:
                from extraction.two_pass import two_pass_extract
                result = two_pass_extract(
                    source=ds.doi,
                    strategy=strategy,
                    extract_client=client,
                    pages_per_chunk=config.pages_per_chunk,
                    output_format=config.output_format,
                    normalize=normalize,
                )
            else:
                result = extract_guideline(
                    source=ds.doi,
                    strategy=strategy,
                    client=client,
                    pages_per_chunk=config.pages_per_chunk,
                    output_format=config.output_format,
                    normalize=normalize,
                    verify=config.verify,
                    verify_threshold=config.verify_threshold,
                    post_filter=config.post_filter,
                    filter_model=config.filter_model,
                )
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            continue
        extract_elapsed = time.time() - start

        # Evaluate
        try:
            eval_result = evaluate_extraction(
                extracted_df=result.recommendations_df,
                gt_df=ds.ground_truth_df,
                grading_scheme=ds.grading_scheme,
                similarity_model=similarity_model,
                similarity_threshold=0.65,
            )
        except Exception as e:
            print(f"EVAL ERROR: {e}", flush=True)
            continue

        entry = BenchmarkEntry(
            model=DEFAULT_MODEL,
            strategy=strategy_name,
            dataset_name=ds.dataset_name,
            guideline_key=ds.key,
            n_gt=len(ds.ground_truth_df),
            n_extracted=len(result.recommendations_df),
            precision=eval_result.precision,
            recall=eval_result.recall,
            f1=eval_result.f1,
            grade_accuracy=eval_result.grade_accuracy,
            level_accuracy=eval_result.level_accuracy,
            combined_accuracy=eval_result.combined_accuracy,
            mean_similarity=eval_result.mean_similarity,
            extraction_time_sec=extract_elapsed,
        )
        entries.append(entry)
        _append_entry(config.name, entry)

        print(f"F1={entry.f1:.2f} P={entry.precision:.2f} R={entry.recall:.2f} "
              f"Grade={entry.grade_accuracy:.2f} [{extract_elapsed:.0f}s]", flush=True)

    return entries


def compute_deltas(baseline_entries, variant_entries):
    """Compare variant vs baseline per guideline. Returns list of dicts."""
    base_map = {e.guideline_key: e for e in baseline_entries}
    deltas = []
    for v in variant_entries:
        b = base_map.get(v.guideline_key)
        if b is None:
            continue
        deltas.append({
            "guideline_key": v.guideline_key,
            "dataset_name": v.dataset_name,
            "base_f1": b.f1,
            "var_f1": v.f1,
            "delta_f1": v.f1 - b.f1,
            "base_precision": b.precision,
            "var_precision": v.precision,
            "delta_precision": v.precision - b.precision,
            "base_recall": b.recall,
            "var_recall": v.recall,
            "delta_recall": v.recall - b.recall,
            "base_grade": b.grade_accuracy,
            "var_grade": v.grade_accuracy,
            "delta_grade": v.grade_accuracy - b.grade_accuracy,
        })
    return deltas


def _avg(values):
    return sum(values) / len(values) if values else 0.0


def generate_summary(all_results, test_configs):
    """Generate markdown summary report."""
    lines = []
    lines.append("# A/B Test Results (15 datasets, 2026-03-11)\n")

    baseline_entries = all_results.get("baseline", [])
    if not baseline_entries:
        lines.append("**ERROR: No baseline results found.**\n")
        return "\n".join(lines)

    n_datasets = len(baseline_entries)
    base_map = {e.guideline_key: e for e in baseline_entries}

    # Config lookup
    config_map = {c.name: c for c in test_configs}

    # ── Summary table (excluding NI9RV3E7) ──
    lines.append(f"## Summary (excluding NI9RV3E7, {n_datasets} datasets total)\n")
    lines.append("| Test | Avg F1 | \u0394F1 | Avg P | Avg R | Avg Grade | "
                 "Prior Conclusion | Validated? |")
    lines.append("|------|--------|-----|-------|-------|-----------|"
                 "-----------------|------------|")

    for test_name in ["baseline"] + [c.name for c in test_configs if c.name != "baseline"]:
        entries = all_results.get(test_name, [])
        if not entries:
            continue
        cfg = config_map.get(test_name)

        excl = [e for e in entries if e.guideline_key != NI9RV3E7_KEY]
        if not excl:
            continue
        avg_f1 = _avg([e.f1 for e in excl])
        avg_p = _avg([e.precision for e in excl])
        avg_r = _avg([e.recall for e in excl])
        avg_grade = _avg([e.grade_accuracy for e in excl])

        if test_name == "baseline":
            delta_str = "—"
            prior = "—"
            validated = "—"
        else:
            base_excl = [e for e in baseline_entries if e.guideline_key != NI9RV3E7_KEY]
            base_avg_f1 = _avg([e.f1 for e in base_excl])
            delta = avg_f1 - base_avg_f1
            delta_str = f"{delta:+.3f}"
            prior = cfg.prior_conclusion if cfg else ""
            # Validated if delta direction matches prior conclusion
            validated = "—"
            if "worse" in prior.lower() or "hurt" in prior.lower() or "-" in prior:
                validated = "\u2713" if delta < -0.01 else "\u2717"
            elif "neutral" in prior.lower() or "zero" in prior.lower():
                validated = "\u2713" if abs(delta) < 0.02 else "\u2717"
            elif "helps" in prior.lower() or "positive" in prior.lower():
                validated = "\u2713" if delta > 0.01 else "\u2717"

        lines.append(f"| {test_name} | {avg_f1:.3f} | {delta_str} | {avg_p:.3f} | "
                     f"{avg_r:.3f} | {avg_grade:.3f} | {prior} | {validated} |")

    # ── Summary including NI9RV3E7 ──
    lines.append(f"\n## Summary (all {n_datasets} datasets)\n")
    lines.append("| Test | Avg F1 | Avg P | Avg R | Avg Grade |")
    lines.append("|------|--------|-------|-------|-----------|")

    for test_name in ["baseline"] + [c.name for c in test_configs if c.name != "baseline"]:
        entries = all_results.get(test_name, [])
        if not entries:
            continue
        avg_f1 = _avg([e.f1 for e in entries])
        avg_p = _avg([e.precision for e in entries])
        avg_r = _avg([e.recall for e in entries])
        avg_grade = _avg([e.grade_accuracy for e in entries])
        lines.append(f"| {test_name} | {avg_f1:.3f} | {avg_p:.3f} | {avg_r:.3f} | {avg_grade:.3f} |")

    # ── Per-guideline detail for each test ──
    for test_name in ["baseline"] + [c.name for c in test_configs if c.name != "baseline"]:
        entries = all_results.get(test_name, [])
        if not entries or test_name == "baseline":
            continue

        deltas = compute_deltas(baseline_entries, entries)
        if not deltas:
            continue

        lines.append(f"\n## Per-Guideline Detail: {test_name}\n")
        lines.append("| Guideline | Base F1 | Var F1 | \u0394F1 | Base Grade | Var Grade | \u0394Grade |")
        lines.append("|-----------|---------|--------|-----|------------|-----------|--------|")

        for d in sorted(deltas, key=lambda x: x["guideline_key"]):
            lines.append(
                f"| {d['dataset_name']}:{d['guideline_key']} "
                f"| {d['base_f1']:.3f} | {d['var_f1']:.3f} | {d['delta_f1']:+.3f} "
                f"| {d['base_grade']:.3f} | {d['var_grade']:.3f} | {d['delta_grade']:+.3f} |"
            )

    # ── Baseline per-guideline ──
    lines.append("\n## Baseline Per-Guideline Detail\n")
    lines.append("| Guideline | Scheme | F1 | P | R | Grade | Level | GT Recs | Extracted | Time |")
    lines.append("|-----------|--------|------|------|------|-------|-------|---------|-----------|------|")
    for e in sorted(baseline_entries, key=lambda x: x.guideline_key):
        lines.append(
            f"| {e.dataset_name}:{e.guideline_key} | — "
            f"| {e.f1:.3f} | {e.precision:.3f} | {e.recall:.3f} "
            f"| {e.grade_accuracy:.3f} | {e.level_accuracy:.3f} "
            f"| {e.n_gt} | {e.n_extracted} | {e.extraction_time_sec:.0f}s |"
        )

    return "\n".join(lines)


def main():
    # ── Parse CLI ──
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    skip_low = "--skip-low-priority" in args
    skip_ensemble = "--skip-ensemble" in args

    only_tests = None
    if "--only" in args:
        idx = args.index("--only")
        only_tests = args[idx + 1].split(",")

    # ── Select tests ──
    tests = list(AB_TESTS)
    if only_tests:
        tests = [t for t in tests if t.name in only_tests]
    elif skip_low:
        tests = [t for t in tests if t.priority <= 6]
    if skip_ensemble:
        tests = [t for t in tests if t.name != "ensemble"]

    # ── Dry run ──
    if dry_run:
        print("DRY RUN — would execute these tests:\n")
        for t in tests:
            print(f"  [{t.priority}] {t.name}: {t.description}")
        print(f"\nTotal: {len(tests)} tests")
        return

    # ── Setup ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading datasets...", flush=True)
    all_ds = load_all_datasets()
    available = filter_available_datasets(all_ds)
    print(f"Available: {len(available)}/{len(all_ds)} datasets", flush=True)
    for ds in available:
        print(f"  {ds.dataset_name}/{ds.key}: {len(ds.ground_truth_df)} GT recs", flush=True)

    print("\nLoading similarity model...", flush=True)
    sim_model = _BioLORDSimilarityModel()
    print("Similarity model ready.\n", flush=True)

    # ── Run tests ──
    all_results = {}
    total_start = time.time()

    # Ensure baseline runs first
    test_order = sorted(tests, key=lambda t: t.priority)

    for config in test_order:
        test_start = time.time()
        print(f"\n{'='*60}", flush=True)
        print(f"A/B TEST: {config.name} — {config.description}", flush=True)
        print(f"{'='*60}", flush=True)

        entries = run_single_test(config, available, sim_model)
        all_results[config.name] = entries

        test_elapsed = time.time() - test_start
        n_done = len(entries)
        if entries:
            excl = [e for e in entries if e.guideline_key != NI9RV3E7_KEY]
            avg_f1 = _avg([e.f1 for e in excl]) if excl else 0
            print(f"\n[A/B: {config.name}] Done: {n_done} guidelines, "
                  f"avg F1={avg_f1:.3f} (excl NI9RV3E7), {test_elapsed:.0f}s total",
                  flush=True)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}", flush=True)
    print(f"All tests complete in {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)", flush=True)

    # ── Generate summary ──
    summary = generate_summary(all_results, test_order)
    summary_path = os.path.join(OUTPUT_DIR, "summary.md")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"\nSummary written to {summary_path}", flush=True)

    # Also print summary to stdout
    print("\n" + summary, flush=True)


if __name__ == "__main__":
    main()
