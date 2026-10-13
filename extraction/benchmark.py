"""Benchmark extraction across models, strategies, and datasets."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from evaluation.evaluate import evaluate_extraction
from evaluation.grading import GradingScheme
from extraction.datasets import GuidelineDataset, load_all_datasets, get_few_shot_examples
from extraction.extractor import extract_guideline
from extraction.llm_client import OllamaClient, AVAILABLE_MODELS
from extraction.pdf_loader import _doi_to_filename, _DEFAULT_PDF_DIR
from extraction.prompts import PromptStrategy

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from similarity_evaluation.similarity_models import SimilarityModel


class _BioLORDSimilarityModel:
    """Lightweight BioLORD similarity model using sentence-transformers directly.

    Avoids importing similarity_evaluation.similarity_models which requires spacy.
    """

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.name = "FremyCompany/BioLORD-2023"
        self._model = SentenceTransformer(self.name)

    def compute_similarity(self, text1: str, text2: str) -> float:
        from sentence_transformers import util
        embs = self._model.encode([text1, text2])
        return float(util.cos_sim(embs[0], embs[1])[0][0])

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> "np.ndarray":
        """Encode a list of texts into normalized embeddings.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim), L2-normalized.
        """
        import numpy as np
        embs = self._model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        embs = np.array(embs)
        # L2-normalize for cosine similarity via dot product
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return embs / norms


MODELS = list(AVAILABLE_MODELS.keys())
STRATEGIES = ["zero_shot", "few_shot"]


def filter_available_datasets(
    datasets: list[GuidelineDataset],
    pdf_dir: str = _DEFAULT_PDF_DIR,
) -> list[GuidelineDataset]:
    """Filter datasets to only those with PDFs available locally."""
    import os
    available = []
    for ds in datasets:
        path = os.path.join(pdf_dir, _doi_to_filename(ds.doi))
        if os.path.isfile(path):
            available.append(ds)
    return available


@dataclass
class BenchmarkEntry:
    """Single benchmark result for one model × strategy × guideline combination."""
    model: str
    strategy: str
    dataset_name: str
    guideline_key: str
    n_gt: int
    n_extracted: int
    precision: float
    recall: float
    f1: float
    grade_accuracy: float
    level_accuracy: float
    combined_accuracy: float
    mean_similarity: float
    extraction_time_sec: float


def _build_strategy(
    name: str,
    scheme: GradingScheme,
    guideline_key: str,
) -> PromptStrategy:
    """Build a PromptStrategy, loading few-shot examples if needed."""
    examples = []
    if name == "few_shot":
        examples = get_few_shot_examples(scheme, n_examples=3, exclude_key=guideline_key)
    return PromptStrategy(name=name, scheme=scheme, examples=examples)


def run_full_benchmark(
    datasets: list[GuidelineDataset] | None = None,
    models: list[str] | None = None,
    strategies: list[str] | None = None,
    similarity_model: SimilarityModel | None = None,
    similarity_threshold: float = 0.65,
) -> pd.DataFrame:
    """Run extraction + evaluation for all model × strategy × dataset combinations.

    Args:
        datasets: GuidelineDatasets to benchmark. Defaults to load_all_datasets().
        models: Model names to benchmark. Defaults to MODELS.
        strategies: Strategy names. Defaults to STRATEGIES.
        similarity_model: Model for matching extracted vs GT recommendations.
        similarity_threshold: Lower threshold (0.65) since LLM extractions may paraphrase.

    Returns:
        DataFrame with one row per benchmark entry.
    """
    if datasets is None:
        datasets = load_all_datasets()
    if models is None:
        models = MODELS
    if strategies is None:
        strategies = STRATEGIES

    if similarity_model is None:
        try:
            from similarity_evaluation.similarity_models import get_similarity_model
            similarity_model = get_similarity_model()
        except ImportError:
            similarity_model = _BioLORDSimilarityModel()

    entries = []

    for model_name in models:
        client = OllamaClient(model=model_name)
        for strategy_name in strategies:
            for ds in datasets:
                print(f"[Benchmark] {model_name} / {strategy_name} / {ds.dataset_name}:{ds.key}")

                strategy = _build_strategy(strategy_name, ds.grading_scheme, ds.key)

                start = time.time()
                try:
                    result = extract_guideline(
                        source=ds.doi,
                        strategy=strategy,
                        client=client,
                    )
                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue
                elapsed = time.time() - start

                # Evaluate against ground truth
                try:
                    eval_result = evaluate_extraction(
                        extracted_df=result.recommendations_df,
                        gt_df=ds.ground_truth_df,
                        grading_scheme=ds.grading_scheme,
                        similarity_model=similarity_model,
                        similarity_threshold=similarity_threshold,
                    )
                except Exception as e:
                    print(f"  EVAL ERROR: {e}")
                    continue

                entry = BenchmarkEntry(
                    model=model_name,
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
                    extraction_time_sec=elapsed,
                )
                entries.append(entry)
                print(f"  P={entry.precision:.2f} R={entry.recall:.2f} F1={entry.f1:.2f} "
                      f"Grade={entry.grade_accuracy:.2f} Level={entry.level_accuracy:.2f}")

    if not entries:
        return pd.DataFrame()

    return pd.DataFrame([vars(e) for e in entries])


def print_benchmark_summary(results_df: pd.DataFrame) -> None:
    """Print a summary table of benchmark results grouped by (model, strategy)."""
    if results_df.empty:
        print("No benchmark results to summarize.")
        return

    metrics = ["precision", "recall", "f1", "grade_accuracy", "level_accuracy",
               "combined_accuracy", "mean_similarity", "extraction_time_sec"]

    summary = results_df.groupby(["model", "strategy"])[metrics].mean()
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY (mean across all guidelines)")
    print("=" * 80)
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
    print("=" * 80)

    # Per-dataset breakdown
    for ds_name in results_df["dataset_name"].unique():
        ds_results = results_df[results_df["dataset_name"] == ds_name]
        ds_summary = ds_results.groupby(["model", "strategy"])[metrics].mean()
        print(f"\n--- {ds_name} ---")
        print(ds_summary.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    all_datasets = load_all_datasets()
    available = filter_available_datasets(all_datasets)
    print(f"Available datasets: {len(available)}/{len(all_datasets)}")
    for ds in available:
        print(f"  {ds.dataset_name}/{ds.key}: {len(ds.ground_truth_df)} GT recommendations")

    if not available:
        print("No PDFs available. Download them first.")
    else:
        results = run_full_benchmark(datasets=available)
        print_benchmark_summary(results)
        results.to_csv("benchmark_results.csv", index=False)
        print("\nResults saved to benchmark_results.csv")
