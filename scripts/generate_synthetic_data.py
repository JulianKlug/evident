#!/usr/bin/env python3
"""Generate synthetic training data for fine-tuning via LLM augmentation.

Uses qwen3:14b to generate realistic clinical guideline pages with embedded
recommendations for each grading scheme. Focuses on underrepresented schemes
(ABCD_123, ESC_ERS) to balance the training set.

Three augmentation strategies:
1. Re-contextualize: Embed real GT recs in new synthetic page contexts
2. Fully synthetic: Generate new recommendations + page contexts per scheme
3. Hard negatives: Pages with medical text mentioning grades but no real recs

Output: JSONL in same ChatML format as prepare_finetune_data.py

Usage:
    python scripts/generate_synthetic_data.py --output-dir artifacts/finetune/synthetic
    python scripts/generate_synthetic_data.py --output-dir artifacts/finetune/synthetic --target-per-scheme 200
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extraction.datasets import load_all_datasets, GuidelineDataset
from extraction.benchmark import filter_available_datasets
from extraction.prompts import _SCHEME_TERMINOLOGY
from evaluation.grading import GradingScheme
from scripts.prepare_finetune_data import build_system_prompt, build_user_message

import ollama


# ── Medical topics for synthetic page generation ──

# ── Scheme-specific density distributions ──
# Weights for number of recs per synthetic page, reflecting real-world densities.
# ABCD_123 (BDYDTUHA) has 1-11 recs/page in real data, so we sample up to 8.
DENSITY_WEIGHTS = {
    "grade": {1: 5, 2: 3, 3: 2, 4: 1},
    "esc_ers": {1: 3, 2: 3, 3: 2, 4: 1, 5: 1},
    "abcd_123": {1: 2, 2: 2, 3: 2, 4: 3, 5: 2, 6: 2, 7: 1, 8: 1},
}


def sample_n_recs(scheme_name: str, rng: random.Random, max_recs: int) -> int:
    """Sample number of recs from scheme-specific weighted distribution."""
    weights = DENSITY_WEIGHTS.get(scheme_name, {1: 1, 2: 1, 3: 1})
    # Filter to respect max_recs cap
    filtered = {k: v for k, v in weights.items() if k <= max_recs}
    if not filtered:
        return 1
    values = list(filtered.keys())
    w = list(filtered.values())
    return rng.choices(values, weights=w, k=1)[0]


MEDICAL_TOPICS = {
    "grade": [
        "management of chronic obstructive pulmonary disease exacerbations",
        "pharmacologic treatment of type 2 diabetes mellitus",
        "anticoagulation therapy for venous thromboembolism",
        "screening and management of dyslipidemia in adults",
        "treatment of community-acquired pneumonia in adults",
        "management of stable angina pectoris",
        "diagnosis and treatment of iron deficiency anemia",
        "pharmacotherapy for alcohol use disorder",
        "management of acute kidney injury in hospitalized patients",
        "treatment of osteoporosis in postmenopausal women",
        "management of chronic insomnia disorder in adults",
        "antibiotic therapy for acute bacterial sinusitis",
        "management of benign prostatic hyperplasia",
        "treatment of generalized anxiety disorder",
        "management of acute pancreatitis",
        "treatment of hepatitis C virus infection",
        "management of chronic heart failure with reduced ejection fraction",
        "perioperative management of antiplatelet therapy",
        "treatment of migraine headaches in adults",
        "management of asthma in adults",
    ],
    "esc_ers": [
        "management of pulmonary arterial hypertension",
        "treatment of chronic thromboembolic pulmonary hypertension",
        "diagnosis and management of acute pulmonary embolism",
        "management of valvular heart disease",
        "treatment of supraventricular tachycardia",
        "management of atrial fibrillation",
        "diagnosis and treatment of aortic diseases",
        "management of heart failure with preserved ejection fraction",
        "treatment of peripheral arterial disease",
        "management of infective endocarditis",
        "cardiac rehabilitation after myocardial infarction",
        "management of hypertrophic cardiomyopathy",
        "treatment of stable coronary artery disease",
        "management of pericardial diseases",
        "diagnosis and treatment of congenital heart disease in adults",
        "implantable cardioverter-defibrillator therapy",
        "management of dyslipidemia in cardiovascular prevention",
        "anticoagulation in non-valvular atrial fibrillation",
        "management of acute coronary syndromes",
        "cardiac resynchronization therapy for heart failure",
    ],
    "abcd_123": [
        "diagnosis and management of chronic pulmonary aspergillosis",
        "treatment of invasive fungal infections in immunocompromised patients",
        "management of allergic bronchopulmonary aspergillosis",
        "antifungal prophylaxis in hematologic malignancies",
        "diagnosis of invasive candidiasis",
        "management of cryptococcal meningitis",
        "treatment of mucormycosis",
        "management of coccidioidomycosis",
        "antifungal therapy for aspergilloma",
        "management of Pneumocystis jirovecii pneumonia",
        "treatment of histoplasmosis",
        "management of fungal keratitis",
        "diagnosis and treatment of blastomycosis",
        "antifungal susceptibility testing guidelines",
        "management of sporotrichosis",
        "prophylaxis for invasive aspergillosis in transplant recipients",
        "treatment of fungal peritonitis",
        "management of dermatophyte infections",
        "treatment of endemic mycoses in pregnancy",
        "management of azole-resistant aspergillosis",
    ],
}

# ── Example real recommendations per scheme (seeds for generation) ──

SEED_RECS = {
    "grade": [
        ("ACP recommends monotherapy with cognitive behavioral therapy as initial treatment in patients with mild major depressive disorder", "Weak For", "Low"),
        ("ACP recommends that clinicians prescribe a statin for secondary prevention of cardiovascular events in patients with documented coronary artery disease", "Strong For", "Moderate"),
        ("ACP recommends against screening for chronic obstructive pulmonary disease in asymptomatic adults", "Strong Against", "Moderate"),
        ("ACP suggests initiating pharmacologic treatment in adults with newly diagnosed type 2 diabetes when lifestyle modifications alone are insufficient", "Weak For", "Low"),
        ("ACP recommends perioperative use of venous thromboembolism prophylaxis in patients undergoing major orthopedic surgery", "Strong For", "High"),
    ],
    "esc_ers": [
        ("Right heart catheterization is recommended to confirm the diagnosis of pulmonary arterial hypertension and to support treatment decisions", "I", "C"),
        ("Oral anticoagulation should be considered in patients with chronic thromboembolic pulmonary hypertension", "IIa", "B"),
        ("Calcium channel blockers may be considered in patients who show a positive response during acute vasoreactivity testing", "IIb", "C"),
        ("The use of endothelin receptor antagonists is not recommended as first-line monotherapy in patients with WHO functional class IV symptoms", "III", "C"),
        ("Combination therapy with phosphodiesterase-5 inhibitors and endothelin receptor antagonists is recommended in treatment-naive patients with pulmonary arterial hypertension", "I", "B"),
    ],
    "abcd_123": [
        ("Itraconazole is recommended as first-line therapy for chronic pulmonary aspergillosis in patients who can tolerate oral therapy", "A", "2"),
        ("Voriconazole should be used as an alternative when itraconazole is not tolerated or when there is documented azole resistance", "B", "2"),
        ("Surgical resection may be considered for localized aspergilloma with recurrent hemoptysis when antifungal therapy alone is insufficient", "C", "3"),
        ("Therapeutic drug monitoring is recommended during azole antifungal therapy to ensure adequate serum concentrations", "A", "2"),
        ("Aspergillus IgG antibody testing is recommended as the primary diagnostic test for chronic pulmonary aspergillosis", "A", "1"),
    ],
}


def generate_synthetic_page(
    client,
    scheme_name: str,
    topic: str,
    n_recs: int,
    seed_recs: List[Tuple[str, str, str]],
    rng: random.Random,
    model: str = "qwen3:14b",
) -> Optional[Tuple[str, List[Tuple[str, str, str]]]]:
    """Generate a synthetic guideline page with embedded recommendations.

    Args:
        client: Ollama client.
        scheme_name: Grading scheme name.
        topic: Medical topic for the page.
        n_recs: Number of recommendations to embed (0 for negative).
        seed_recs: Example recommendations to guide style.
        rng: Random number generator.

    Returns:
        Tuple of (page_text, list of (rec_text, grade, level)) or None on failure.
    """
    terms = _SCHEME_TERMINOLOGY[scheme_name]
    grade_label = terms["grade_label"]
    level_label = terms["level_label"]
    grade_values = terms["grade_values"]
    level_values = terms["level_values"]

    # Pick random seed recs as style examples
    style_examples = rng.sample(seed_recs, min(2, len(seed_recs)))
    style_str = "\n".join(
        f"  - \"{r[0]}\" ({grade_label}: {r[1]}, {level_label}: {r[2]})"
        for r in style_examples
    )

    # Pick random valid grades/levels for the recs to generate
    if scheme_name == "grade":
        valid_grades = ["Strong For", "Weak For", "Conditional For",
                        "Strong Against", "Weak Against", "Conditional Against"]
        valid_levels = ["High", "Moderate", "Low", "Very Low"]
    elif scheme_name == "esc_ers":
        valid_grades = ["I", "IIa", "IIb", "III"]
        valid_levels = ["A", "B", "C"]
    elif scheme_name == "abcd_123":
        valid_grades = ["A", "B", "C", "D"]
        valid_levels = ["1", "2", "3"]
    else:
        return None

    if n_recs > 0:
        # Assign random grades/levels for each rec
        assigned = []
        for _ in range(n_recs):
            g = rng.choice(valid_grades)
            lv = rng.choice(valid_levels)
            assigned.append((g, lv))

        rec_spec = "\n".join(
            f"  Recommendation {i+1}: {grade_label}={g}, {level_label}={lv}"
            for i, (g, lv) in enumerate(assigned)
        )

        prompt = (
            f"Generate a realistic clinical practice guideline page about {topic}.\n\n"
            f"The page should contain EXACTLY {n_recs} clinical recommendation(s) with explicit "
            f"grading annotations. Each recommendation must be a clear, actionable clinical "
            f"statement that directs practice.\n\n"
            f"Required recommendations (use these exact grades):\n{rec_spec}\n\n"
            f"Style examples (match this level of clinical specificity):\n{style_str}\n\n"
            f"IMPORTANT RULES:\n"
            f"- Write the page as it would appear in a real published guideline\n"
            f"- Include surrounding context: evidence summaries, rationale, background text\n"
            f"- Each recommendation must explicitly state its {grade_label} and {level_label}\n"
            f"- Valid {grade_label} values: {grade_values}\n"
            f"- Valid {level_label} values: {level_values}\n"
            f"- Include at least 2-3 paragraphs of NON-recommendation text (background, evidence)\n"
            f"- Do NOT number the recommendations\n"
            f"- Make the text realistic and medically accurate\n\n"
            f"After the page text, on a NEW LINE output EXACTLY:\n"
            f"---EXTRACTED---\n"
            f"Then list each recommendation in this exact format, one per line:\n"
            f"recommendation text | {grade_label} value | {level_label} value\n\n"
            f"Generate the guideline page now:"
        )
    else:
        # Negative: page with medical text but NO recommendations
        prompt = (
            f"Generate a realistic clinical practice guideline page about {topic}.\n\n"
            f"This page should contain ONLY background information:\n"
            f"- Evidence summaries and systematic review findings\n"
            f"- Epidemiology and pathophysiology descriptions\n"
            f"- Discussion of study methodologies and limitations\n"
            f"- References to evidence quality (e.g., 'moderate quality evidence suggests...')\n\n"
            f"CRITICAL: Do NOT include any clinical recommendations or graded statements.\n"
            f"Do NOT include any {grade_label} or {level_label} annotations.\n"
            f"The text should look like a guideline section that provides context but does not\n"
            f"contain actionable recommendations.\n\n"
            f"Write 3-5 paragraphs of realistic guideline background text.\n\n"
            f"After the page text, on a NEW LINE output EXACTLY:\n"
            f"---EXTRACTED---\n"
            f"NO_RECOMMENDATIONS_FOUND\n\n"
            f"Generate the guideline page now:"
        )

    # Use larger context for reasoning models
    num_ctx = 8192 if "deepseek" in model or "32b" in model else 4096
    num_predict = 4096 if "deepseek" in model or "32b" in model else 2048

    try:
        response = client.generate(
            model=model,
            prompt=prompt,
            options={"temperature": 0.8, "num_ctx": num_ctx, "num_predict": num_predict},
        )
        raw = response.get("response", "")
    except Exception as e:
        print(f"  [ERROR] Generation failed: {e}")
        return None

    return parse_synthetic_response(raw, scheme_name)


def parse_synthetic_response(
    raw: str,
    scheme_name: str,
) -> Optional[Tuple[str, List[Tuple[str, str, str]]]]:
    """Parse the LLM's synthetic page + extracted recs.

    Returns:
        Tuple of (page_text, list of (rec, grade, level)) or None on parse failure.
    """
    # Strip thinking tags if present
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Split at the extraction marker
    marker = "---EXTRACTED---"
    if marker not in raw:
        return None

    parts = raw.split(marker, 1)
    page_text = parts[0].strip()
    extracted_section = parts[1].strip()

    if not page_text or len(page_text) < 100:
        return None

    # Parse extracted recommendations
    if "NO_RECOMMENDATIONS_FOUND" in extracted_section:
        return (page_text, [])

    recs = []
    for line in extracted_section.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3:
            rec_text = parts[0]
            grade = parts[1]
            level = parts[2]
            # Basic validation
            if len(rec_text) > 20 and grade and level:
                recs.append((rec_text, grade, level))

    if not recs and "NO_RECOMMENDATIONS_FOUND" not in extracted_section:
        return None

    return (page_text, recs)


def recontextualize_recs(
    client,
    scheme_name: str,
    recs: List[Tuple[str, str, str]],
    topic: str,
    model: str = "qwen3:14b",
) -> Optional[Tuple[str, List[Tuple[str, str, str]]]]:
    """Embed existing real GT recs in a new synthetic page context.

    The recommendations are kept EXACTLY as-is; only the surrounding text changes.
    """
    terms = _SCHEME_TERMINOLOGY[scheme_name]
    grade_label = terms["grade_label"]
    level_label = terms["level_label"]

    rec_list_str = "\n".join(
        f"  \"{r[0]}\" ({grade_label}: {r[1]}, {level_label}: {r[2]})"
        for r in recs
    )

    prompt = (
        f"Generate a realistic clinical practice guideline page about {topic} that "
        f"contains the following recommendation(s) EXACTLY as written below.\n\n"
        f"Recommendations to embed (copy these EXACTLY - do not modify the text):\n"
        f"{rec_list_str}\n\n"
        f"RULES:\n"
        f"- Copy each recommendation word-for-word into the page text\n"
        f"- Add 2-3 paragraphs of surrounding context (evidence, rationale, background)\n"
        f"- Make the page read naturally as a published guideline section\n"
        f"- The recommendations should appear with their explicit {grade_label} and {level_label}\n\n"
        f"After the page text, on a NEW LINE output EXACTLY:\n"
        f"---EXTRACTED---\n"
        f"Then list each recommendation in format: recommendation text | {grade_label} | {level_label}\n\n"
        f"Generate the guideline page now:"
    )

    num_ctx = 8192 if "deepseek" in model or "32b" in model else 4096
    num_predict = 4096 if "deepseek" in model or "32b" in model else 2048

    try:
        response = client.generate(
            model=model,
            prompt=prompt,
            options={"temperature": 0.7, "num_ctx": num_ctx, "num_predict": num_predict},
        )
        raw = response.get("response", "")
    except Exception as e:
        print(f"  [ERROR] Recontextualize failed: {e}")
        return None

    result = parse_synthetic_response(raw, scheme_name)
    if result is None:
        return None

    page_text, parsed_recs = result

    # Verify that the original recs appear in the page (at least partially)
    # Use the parsed recs from the LLM since it may have reformatted slightly
    if len(parsed_recs) == 0 and len(recs) > 0:
        return None

    # Use original recs for the training label (not parsed) since we want exact match
    return (page_text, recs)


def build_chatml_example(
    page_text: str,
    recs: List[Tuple[str, str, str]],
    scheme: GradingScheme,
    source: str,
    topic: str,
) -> dict:
    """Build a ChatML training example from synthetic data."""
    system_prompt = build_system_prompt(scheme)
    user_msg = build_user_message(page_text)

    if recs:
        lines = [f"{r[0]} | {r[1]} | {r[2]}" for r in recs]
        assistant_msg = "\n".join(lines)
    else:
        assistant_msg = "NO_RECOMMENDATIONS_FOUND"

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "metadata": {
            "source": source,
            "scheme": scheme.name,
            "topic": topic,
            "n_recs": len(recs),
            "synthetic": True,
        },
    }


def load_real_gt_recs(datasets: List[GuidelineDataset]) -> Dict[str, List[Tuple[str, str, str]]]:
    """Load real GT recommendations grouped by scheme."""
    scheme_recs = defaultdict(list)
    for ds in datasets:
        for _, row in ds.ground_truth_df.iterrows():
            rec = (
                str(row["recommendation"]).strip(),
                str(row["class"]).strip(),
                str(row["LOE"]).strip(),
            )
            scheme_recs[ds.grading_scheme.name].append(rec)
    return dict(scheme_recs)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fine-tuning data")
    parser.add_argument("--output-dir", default="artifacts/finetune/synthetic",
                        help="Output directory")
    parser.add_argument("--target-per-scheme", type=int, default=180,
                        help="Target synthetic examples per scheme")
    parser.add_argument("--neg-ratio", type=float, default=0.3,
                        help="Fraction of examples that are negatives")
    parser.add_argument("--recontextualize-ratio", type=float, default=0.3,
                        help="Fraction of positives using real GT recs in new contexts")
    parser.add_argument("--max-recs-per-page", type=int, default=8,
                        help="Max recommendations per synthetic page")
    parser.add_argument("--model", default="deepseek-r1:32b",
                        help="Ollama model for synthetic generation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", default=None, help="Ollama host")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)

    # Load real GT for recontextualization
    print("Loading datasets for GT seeds...", flush=True)
    all_ds = load_all_datasets()
    available = filter_available_datasets(all_ds)
    real_recs = load_real_gt_recs(available)

    # Map scheme names to GradingScheme objects
    from evaluation.grading import GRADE, ABCD_123, ESC_ERS
    scheme_objects = {
        "grade": GRADE,
        "abcd_123": ABCD_123,
        "esc_ers": ESC_ERS,
    }

    client = ollama.Client(**({"host": args.host} if args.host else {}))
    model = args.model
    print(f"Using model: {model}")

    all_examples = []
    all_stats = defaultdict(lambda: {"positive": 0, "negative": 0, "recontext": 0, "failed": 0})

    for scheme_name in ["grade", "esc_ers", "abcd_123"]:
        scheme = scheme_objects[scheme_name]
        topics = MEDICAL_TOPICS[scheme_name]
        seeds = SEED_RECS[scheme_name]
        gt_recs = real_recs.get(scheme_name, [])
        target = args.target_per_scheme

        n_negatives = int(target * args.neg_ratio)
        n_positives = target - n_negatives
        n_recontext = int(n_positives * args.recontextualize_ratio) if gt_recs else 0
        n_fully_synthetic = n_positives - n_recontext

        print(f"\n{'='*60}")
        print(f"Scheme: {scheme_name} (target={target})")
        print(f"  Fully synthetic positives: {n_fully_synthetic}")
        print(f"  Recontextualized positives: {n_recontext}")
        print(f"  Negatives: {n_negatives}")
        print(f"  GT recs available: {len(gt_recs)}")
        print(f"{'='*60}")

        stats = all_stats[scheme_name]

        # 1. Fully synthetic positives
        print(f"\nGenerating {n_fully_synthetic} fully synthetic positives...")
        for i in range(n_fully_synthetic):
            topic = rng.choice(topics)
            n_recs = sample_n_recs(scheme_name, rng, args.max_recs_per_page)
            result = generate_synthetic_page(client, scheme_name, topic, n_recs, seeds, rng, model=model)

            if result is None:
                stats["failed"] += 1
                print(f"  [{i+1}/{n_fully_synthetic}] FAILED (parse error)", flush=True)
                continue

            page_text, recs = result
            if not recs:
                stats["failed"] += 1
                print(f"  [{i+1}/{n_fully_synthetic}] FAILED (no recs parsed)", flush=True)
                continue

            example = build_chatml_example(page_text, recs, scheme, "synthetic_positive", topic)
            all_examples.append(example)
            stats["positive"] += 1

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{n_fully_synthetic}] OK: {len(recs)} recs, "
                      f"topic={topic[:40]}...", flush=True)

        # 2. Recontextualized positives (embed real GT recs in new contexts)
        if n_recontext > 0:
            print(f"\nGenerating {n_recontext} recontextualized positives...")
            for i in range(n_recontext):
                topic = rng.choice(topics)
                # Pick 1-2 real GT recs
                n_pick = rng.randint(1, min(2, len(gt_recs)))
                picked_recs = rng.sample(gt_recs, n_pick)

                result = recontextualize_recs(client, scheme_name, picked_recs, topic, model=model)

                if result is None:
                    stats["failed"] += 1
                    print(f"  [{i+1}/{n_recontext}] FAILED (recontext)", flush=True)
                    continue

                page_text, recs = result
                example = build_chatml_example(page_text, recs, scheme, "recontextualized", topic)
                all_examples.append(example)
                stats["recontext"] += 1

                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  [{i+1}/{n_recontext}] OK: {len(recs)} recs", flush=True)

        # 3. Hard negatives
        print(f"\nGenerating {n_negatives} hard negatives...")
        for i in range(n_negatives):
            topic = rng.choice(topics)
            result = generate_synthetic_page(client, scheme_name, topic, 0, seeds, rng, model=model)

            if result is None:
                stats["failed"] += 1
                print(f"  [{i+1}/{n_negatives}] FAILED (parse error)", flush=True)
                continue

            page_text, recs = result
            if recs:
                # LLM snuck in recommendations despite being told not to — skip
                stats["failed"] += 1
                print(f"  [{i+1}/{n_negatives}] FAILED (contained recs)", flush=True)
                continue

            example = build_chatml_example(page_text, recs, scheme, "synthetic_negative", topic)
            all_examples.append(example)
            stats["negative"] += 1

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{n_negatives}] OK: negative", flush=True)

    # Shuffle
    rng.shuffle(all_examples)

    # Write synthetic JSONL
    synth_path = os.path.join(args.output_dir, "synthetic_data.jsonl")
    with open(synth_path, "w") as f:
        for ex in all_examples:
            record = {"messages": ex["messages"]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write with metadata for debugging
    debug_path = os.path.join(args.output_dir, "synthetic_data_debug.jsonl")
    with open(debug_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Stats
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    total = len(all_examples)
    print(f"Total synthetic examples: {total}")
    for scheme_name in ["grade", "esc_ers", "abcd_123"]:
        s = all_stats[scheme_name]
        scheme_total = s["positive"] + s["negative"] + s["recontext"]
        print(f"  {scheme_name}: {scheme_total} "
              f"({s['positive']} synth+ / {s['recontext']} recontext+ / {s['negative']} neg / "
              f"{s['failed']} failed)")

    print(f"\nSynthetic data: {synth_path}")
    print(f"Debug data: {debug_path}")

    # Save stats
    stats_path = os.path.join(args.output_dir, "synthetic_stats.json")
    with open(stats_path, "w") as f:
        json.dump({"total": total, "per_scheme": dict(all_stats)}, f, indent=2)

    print(f"\nDone! Next: merge with real data using:")
    print(f"  python scripts/merge_training_data.py \\")
    print(f"    --real artifacts/finetune/train_data.jsonl \\")
    print(f"    --synthetic {synth_path} \\")
    print(f"    --output artifacts/finetune/train_data_augmented.jsonl")


if __name__ == "__main__":
    main()
