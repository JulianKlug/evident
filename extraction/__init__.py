from extraction.response_parser import parse_llm_response
from extraction.prompts import PromptStrategy, build_prompt
from extraction.llm_client import OllamaClient, LLMResponse, AVAILABLE_MODELS
from extraction.pdf_loader import load_pdf_pages, PDFPage
from extraction.deduplication import deduplicate_recommendations
from extraction.datasets import (
    GuidelineDataset,
    load_acp_datasets,
    load_ers_datasets,
    load_icu_datasets,
    load_all_datasets,
    get_few_shot_examples,
)
from extraction.extractor import ExtractionResult, extract_guideline
from extraction.benchmark import run_full_benchmark, print_benchmark_summary
