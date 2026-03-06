"""Ground truth dataset loading for ACP and ERS guidelines."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import pandas as pd

from evaluation.grading import GradingScheme, GRADE, ABCD_123, ESC_ERS


_DATA_ROOT = "/mnt/data1/klug/datasets/evidence_extraction"
_ACP_DIR = os.path.join(_DATA_ROOT, "General internal medicine", "ACP")
_ERS_DIR = os.path.join(_DATA_ROOT, "Pneumology", "ERS_guidelines")


@dataclass
class GuidelineDataset:
    """A single guideline with its ground truth extraction."""
    key: str
    title: str
    doi: str
    ground_truth_df: pd.DataFrame  # columns: recommendation, class, LOE
    grading_scheme: GradingScheme
    dataset_name: str  # "ACP" or "ERS"


def _load_extraction_xlsx(path: str) -> pd.DataFrame:
    """Load an extraction xlsx and return only [recommendation, class, LOE]."""
    df = pd.read_excel(path)
    # Keep only the columns we need
    df = df[["recommendation", "class", "LOE"]].copy()
    # Drop rows where recommendation is NaN
    df = df.dropna(subset=["recommendation"])
    # Normalize class and LOE: strip whitespace, fix case
    df["class"] = df["class"].astype(str).str.strip().str.rstrip(";,.")
    df["LOE"] = df["LOE"].astype(str).str.strip().str.rstrip(";,.")
    return df.reset_index(drop=True)


def load_acp_datasets() -> list[GuidelineDataset]:
    """Load all ACP guideline datasets (GRADE scheme)."""
    metadata_path = os.path.join(_ACP_DIR, "ACP.csv")
    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")

    datasets = []
    for _, row in metadata.iterrows():
        key = row["Key"]
        extraction_path = os.path.join(_ACP_DIR, f"{key}_extraction.xlsx")
        if not os.path.isfile(extraction_path):
            continue
        gt_df = _load_extraction_xlsx(extraction_path)
        # Normalize GT grades/levels (fixes truncation artifacts, etc.)
        gt_df["class"] = gt_df["class"].apply(
            lambda x: GRADE.normalize_grade(x) or x
        )
        gt_df["LOE"] = gt_df["LOE"].apply(
            lambda x: GRADE.normalize_level(x) or x
        )
        # Drop rows with invalid/missing values (e.g. '0.0')
        gt_df = gt_df[~gt_df["class"].isin(["0.0", "nan", ""])].reset_index(drop=True)
        datasets.append(GuidelineDataset(
            key=key,
            title=str(row.get("Title", "")),
            doi=str(row.get("DOI", "")),
            ground_truth_df=gt_df,
            grading_scheme=GRADE,
            dataset_name="ACP",
        ))
    return datasets


def _detect_grading_scheme(gt_df: pd.DataFrame) -> GradingScheme:
    """Auto-detect grading scheme from ground truth values."""
    grades = set(gt_df["class"].str.lower().str.strip())
    # GRADE-style: "strong recommendation", "conditional recommendation"
    grade_keywords = {"strong recommendation", "conditional recommendation",
                      "strong recommendation against", "conditional recommendation against"}
    if grades & grade_keywords:
        return GRADE
    # ESC_ERS-style: Roman numerals I, IIa, IIb, III
    esc_keywords = {"i", "iia", "iib", "iii"}
    if grades & esc_keywords:
        return ESC_ERS
    # Default: ABCD_123
    return ABCD_123


# OCR artifacts in NI9RV3E7: 'L' misread for 'I' in Roman numerals
_ESC_ERS_GRADE_FIXES = {
    "ILA": "IIa", "ILB": "IIb", "ILI": "III",
    "LIB": "IIb", "LLA": "IIa", "LLI": "III",
}


def _normalize_ers_gt(gt_df: pd.DataFrame, scheme: GradingScheme) -> pd.DataFrame:
    """Normalize ERS ground truth values based on detected scheme."""
    gt_df = gt_df.copy()
    gt_df["class"] = gt_df["class"].str.upper().str.strip()
    gt_df["class"] = gt_df["class"].str.replace(r"/.*", "", regex=True)

    if scheme is ESC_ERS:
        # Fix OCR artifacts
        gt_df["class"] = gt_df["class"].replace(_ESC_ERS_GRADE_FIXES)
        # Normalize to canonical ESC_ERS values
        gt_df["class"] = gt_df["class"].apply(
            lambda x: ESC_ERS.normalize_grade(x) or x
        )
        gt_df["LOE"] = gt_df["LOE"].apply(
            lambda x: ESC_ERS.normalize_level(x) or x
        )
    elif scheme is GRADE:
        gt_df["class"] = gt_df["class"].apply(
            lambda x: GRADE.normalize_grade(x) or x
        )
        gt_df["LOE"] = gt_df["LOE"].apply(
            lambda x: GRADE.normalize_level(x) or x
        )
    else:
        gt_df["LOE"] = gt_df["LOE"].apply(
            lambda x: ABCD_123.normalize_level(x) or x
        )
    return gt_df


def load_ers_datasets() -> list[GuidelineDataset]:
    """Load all ERS guideline datasets with auto-detected grading scheme."""
    metadata_path = os.path.join(_ERS_DIR, "ERS_guidelines.csv")
    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")

    datasets = []
    for _, row in metadata.iterrows():
        key = row["Key"]
        extraction_path = os.path.join(_ERS_DIR, f"{key}_extraction.xlsx")
        if not os.path.isfile(extraction_path):
            continue
        gt_df = _load_extraction_xlsx(extraction_path)
        scheme = _detect_grading_scheme(gt_df)
        gt_df = _normalize_ers_gt(gt_df, scheme)
        # Drop rows with invalid/missing values
        gt_df = gt_df[~gt_df["class"].isin(["0.0", "nan", ""])].reset_index(drop=True)
        datasets.append(GuidelineDataset(
            key=key,
            title=str(row.get("Title", "")),
            doi=str(row.get("DOI", "")),
            ground_truth_df=gt_df,
            grading_scheme=scheme,
            dataset_name="ERS",
        ))
    return datasets


def load_all_datasets() -> list[GuidelineDataset]:
    """Load all ACP and ERS datasets."""
    return load_acp_datasets() + load_ers_datasets()


def get_few_shot_examples(
    scheme: GradingScheme,
    n_examples: int = 3,
    exclude_key: str | None = None,
    include_source_text: bool = False,
) -> list[dict]:
    """Sample few-shot examples from ground truth of the same grading scheme.

    Excludes the target guideline (by key) to prevent data leakage.

    Args:
        scheme: Grading scheme to match.
        n_examples: Number of examples to sample.
        exclude_key: Guideline key to exclude (prevents leakage).
        include_source_text: If True, include a truncated source_text excerpt
            from the recommendation text to illustrate input→output mapping.
    """
    if scheme.name == "grade":
        datasets = load_acp_datasets()
    elif scheme.name in ("abcd_123", "esc_ers"):
        datasets = [ds for ds in load_ers_datasets() if ds.grading_scheme.name == scheme.name]
    else:
        return []

    # Pool all GT rows, excluding the target guideline
    pool = []
    for ds in datasets:
        if exclude_key and ds.key == exclude_key:
            continue
        for _, row in ds.ground_truth_df.iterrows():
            example = {
                "recommendation": row["recommendation"],
                "class": row["class"],
                "LOE": row["LOE"],
            }
            if include_source_text:
                # Use a truncated version of the recommendation as a source excerpt
                rec_text = str(row["recommendation"])
                example["source_text"] = rec_text[:200] + ("..." if len(rec_text) > 200 else "")
            pool.append(example)

    if not pool:
        return []

    n = min(n_examples, len(pool))
    return random.sample(pool, n)
