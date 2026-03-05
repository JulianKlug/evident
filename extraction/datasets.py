"""Ground truth dataset loading for ACP and ERS guidelines."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import pandas as pd

from evaluation.grading import GradingScheme, GRADE, ABCD_123


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
    df["class"] = df["class"].astype(str).str.strip()
    df["LOE"] = df["LOE"].astype(str).str.strip()
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
        datasets.append(GuidelineDataset(
            key=key,
            title=str(row.get("Title", "")),
            doi=str(row.get("DOI", "")),
            ground_truth_df=gt_df,
            grading_scheme=GRADE,
            dataset_name="ACP",
        ))
    return datasets


def load_ers_datasets() -> list[GuidelineDataset]:
    """Load all ERS guideline datasets (ABCD_123 scheme)."""
    metadata_path = os.path.join(_ERS_DIR, "ERS_guidelines.csv")
    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")

    datasets = []
    for _, row in metadata.iterrows():
        key = row["Key"]
        extraction_path = os.path.join(_ERS_DIR, f"{key}_extraction.xlsx")
        if not os.path.isfile(extraction_path):
            continue
        gt_df = _load_extraction_xlsx(extraction_path)
        # Normalize messy ERS class values: 'a' → 'A', 'A ' → 'A', 'c/d' → 'C'
        gt_df["class"] = gt_df["class"].str.upper().str.strip()
        gt_df["class"] = gt_df["class"].str.replace(r"/.*", "", regex=True)
        # Normalize Roman numeral LOE values via the grading scheme
        gt_df["LOE"] = gt_df["LOE"].apply(
            lambda x: ABCD_123.normalize_level(x) or x
        )
        datasets.append(GuidelineDataset(
            key=key,
            title=str(row.get("Title", "")),
            doi=str(row.get("DOI", "")),
            ground_truth_df=gt_df,
            grading_scheme=ABCD_123,
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
    elif scheme.name == "abcd_123":
        datasets = load_ers_datasets()
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
