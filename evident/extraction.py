"""Adapter over legacy extraction/: frozen config, version identity, one extraction run.

    ExtractorConfig ──probe_artifacts──▶ Artifacts ──┐
          │                                          ▼
          └───────────────────────────────▶ version_for ──▶ ExtractorVersion (id = sha256)

    run(cfg, pdf, doi, models) ──▶ extract_guideline | self_consistency_extract ──▶ ExtractionOutput

The only evident module, with validation, that imports legacy extraction code.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Optional

from evaluation.grading import GRADE
from evident.domain import Category, ExtractorVersion, LlmCall, RawRecommendation, ThinkingMode
from extraction.datasets import get_few_shot_examples, load_acp_datasets, load_icu_datasets
from extraction.extractor import CallRecord, extract_guideline
from extraction.llm_client import OllamaClient, ThinkMode
from extraction.prompts import PromptStrategy, PromptVersion
from extraction.self_consistency import self_consistency_extract

_BIOLORD = "FremyCompany/BioLORD-2023"
_ZERO_SHOT = "zero_shot"
_FEW_SHOT = "few_shot"
_POOL_COLUMNS = ("recommendation", "class", "LOE")


class ExtractionMode(str, Enum):
    SINGLE_PASS = "single_pass"
    SELF_CONSISTENCY = "self_consistency"  # M2 adds TABLE_FIRST


class FewShotMode(str, Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"


class ConsensusMode(str, Enum):
    FIXED = "fixed"
    ADAPTIVE = "adaptive"


class MlFilterMode(str, Enum):
    OFF = "off"
    ON = "on"


_ENUM_FIELDS = {
    "mode": ExtractionMode,
    "thinking": ThinkingMode,
    "few_shot": FewShotMode,
    "sc_consensus": ConsensusMode,
    "ml_filter": MlFilterMode,
}
_THINK_MODES = {ThinkingMode.OFF: ThinkMode.OFF, ThinkingMode.ON: ThinkMode.ON}


class ExtractionError(Exception):
    """Base class for errors raised by this adapter."""


class ArtifactMismatchError(ExtractionError):
    def __init__(self, field: str, stored: Optional[str], live: Optional[str]):
        super().__init__(f"{field} changed since the version was registered: {stored} → {live}")
        self.field = field
        self.stored = stored
        self.live = live


class PdfMissingError(ExtractionError):
    def __init__(self, path: str):
        super().__init__(f"PDF not found: {path}")
        self.path = path


@dataclass(frozen=True)
class ExtractorConfig:
    """Every extraction choice. One config + its artifacts = one ExtractorVersion.

    SC fields are ignored in SINGLE_PASS but still hashed.
    """
    mode: ExtractionMode
    model: str
    thinking: ThinkingMode
    prompt_version: str  # PromptVersion value
    few_shot: FewShotMode
    n_examples: int = 3
    few_shot_seed: int = 0
    seed: int = 0  # base LLM seed; SC sample k uses seed + k
    pages_per_chunk: int = 1
    dedup_threshold: float = 0.9
    sc_samples: int = 3
    sc_temperature: float = 0.3
    sc_consensus: ConsensusMode = ConsensusMode.ADAPTIVE
    sc_consensus_threshold: int = 2
    sc_similarity_threshold: float = 0.85
    ml_filter: MlFilterMode = MlFilterMode.OFF
    ml_filter_threshold: float = 0.3
    classifier_path: str = "artifacts/classifier/rec_classifier.joblib"
    embedding_model: str = _BIOLORD

    def __post_init__(self):
        PromptVersion(self.prompt_version)  # fail on an unknown prompt now, not mid-snapshot

    def to_json(self) -> str:
        return _canonical(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "ExtractorConfig":
        data = json.loads(s)
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown ExtractorConfig field(s): {sorted(unknown)}")
        for name, enum in _ENUM_FIELDS.items():
            if name in data:
                data[name] = enum(data[name])
        return cls(**data)


@dataclass(frozen=True)
class Artifacts:
    """What the live machine has right now; part of the version identity."""
    model_digest: str
    classifier_sha256: Optional[str]
    embedding_revision: str
    few_shot_pool_sha256: Optional[str]


@dataclass(frozen=True)
class Models:
    """Loaded once per process (BioLORD), shared by dedup, SC clustering and the ML filter."""
    similarity: object


@dataclass(frozen=True)
class ExtractionOutput:
    recs: list[RawRecommendation]  # ordinal = row order; raw class/LOE, not normalized
    calls: list[LlmCall]
    n_pages: int
    few_shot: list[dict]  # examples actually used


def probe_artifacts(cfg: ExtractorConfig) -> Artifacts:
    return Artifacts(
        model_digest=OllamaClient(model=cfg.model).model_digest(),
        classifier_sha256=_file_sha256(cfg.classifier_path) if cfg.ml_filter == MlFilterMode.ON else None,
        embedding_revision=_embedding_revision(cfg.embedding_model),
        few_shot_pool_sha256=_few_shot_pool_sha256() if cfg.few_shot == FewShotMode.FEW_SHOT else None,
    )


def version_for(cfg: ExtractorConfig, artifacts: Artifacts) -> ExtractorVersion:
    identity = {"config": json.loads(cfg.to_json()), **asdict(artifacts)}
    return ExtractorVersion(
        id=hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest(),
        config_json=cfg.to_json(),
        model_name=cfg.model,
        model_digest=artifacts.model_digest,
        classifier_sha256=artifacts.classifier_sha256,
        embedding_revision=artifacts.embedding_revision,
    )


def check_artifacts(version: ExtractorVersion, live: Artifacts) -> None:
    """Raise ArtifactMismatchError naming the first artifact that drifted."""
    stored = {
        "model_digest": version.model_digest,
        "classifier_sha256": version.classifier_sha256,
        "embedding_revision": version.embedding_revision,
    }
    for name, value in stored.items():
        if getattr(live, name) != value:
            raise ArtifactMismatchError(name, value, getattr(live, name))

    # The pool hash has no column; with the other fields equal, a different id can only be the pool
    cfg = ExtractorConfig.from_json(version.config_json)
    live_id = version_for(cfg, live).id
    if live_id != version.id:
        raise ArtifactMismatchError("few_shot_pool_sha256", f"(in version {version.id[:12]})",
                                    live.few_shot_pool_sha256)


def load_models(cfg: ExtractorConfig) -> Models:
    if cfg.embedding_model != _BIOLORD:
        raise ValueError(f"Only {_BIOLORD} is supported, not {cfg.embedding_model}")

    from extraction.benchmark import _BioLORDSimilarityModel
    return Models(similarity=_BioLORDSimilarityModel())


def run(cfg: ExtractorConfig, pdf_path: str, target_doi: str, models: Models) -> ExtractionOutput:
    # Guard before any legacy call: load_pdf_pages falls back to a DOI download for missing paths
    if not os.path.isfile(pdf_path):
        raise PdfMissingError(pdf_path)

    examples = []
    if cfg.few_shot == FewShotMode.FEW_SHOT:
        examples = get_few_shot_examples(GRADE, cfg.n_examples, exclude_doi=target_doi, seed=cfg.few_shot_seed)
    strategy = PromptStrategy(
        name=_FEW_SHOT if examples else _ZERO_SHOT, scheme=GRADE, examples=examples,
        prompt_version=PromptVersion(cfg.prompt_version),
    )

    client = OllamaClient(model=cfg.model)
    common = dict(
        strategy=strategy, client=client, pages_per_chunk=cfg.pages_per_chunk, normalize=False,
        ml_filter=cfg.ml_filter == MlFilterMode.ON, classifier_path=cfg.classifier_path,
        ml_similarity_model=models.similarity, ml_filter_threshold=cfg.ml_filter_threshold,
        seed=cfg.seed, think=_THINK_MODES[cfg.thinking],
    )

    if cfg.mode == ExtractionMode.SELF_CONSISTENCY:
        result = self_consistency_extract(
            pdf_path, similarity_model=models.similarity, n_samples=cfg.sc_samples,
            temperature=cfg.sc_temperature, consensus_threshold=cfg.sc_consensus_threshold,
            similarity_threshold=cfg.sc_similarity_threshold,
            adaptive_threshold=cfg.sc_consensus == ConsensusMode.ADAPTIVE, **common,
        )
    else:
        result = extract_guideline(
            pdf_path, dedup_threshold=cfg.dedup_threshold, dedup_model=models.similarity, **common,
        )

    return ExtractionOutput(
        recs=[_to_rec(i, row) for i, row in enumerate(result.recommendations_df.to_dict("records"))],
        calls=[_to_call(r) for r in result.call_records],
        n_pages=result.n_pages,
        few_shot=examples,
    )


def _canonical(data: dict) -> str:
    """Sorted keys, enum values, no whitespace: equal data → equal string."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=lambda e: e.value)


def _to_rec(ordinal: int, row: dict) -> RawRecommendation:
    return RawRecommendation(
        ordinal=ordinal,
        text=str(row["recommendation"]),
        raw_strength=str(row.get("class", "")),
        raw_certainty=str(row.get("LOE", "")),
        raw_category=Category(row.get("category", Category.GRADED.value)),
        page=_optional_int(row.get("page")),
        chunk_id=_optional_int(row.get("chunk_id")),
    )


def _to_call(record: CallRecord) -> LlmCall:
    r = record.response
    return LlmCall(
        chunk_id=record.chunk_id, prompt_sha256=record.prompt_sha256, raw_text=r.raw_text,
        sample_idx=record.sample_idx, seed=record.seed, prompt_tokens=r.prompt_tokens,
        eval_tokens=r.eval_tokens, duration_ms=r.total_duration_ms,
    )


def _optional_int(value) -> Optional[int]:
    # pandas gives numpy ints, or NaN for a missing value
    if value is None or value != value:
        return None
    return int(value)


def _file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _embedding_revision(repo_id: str) -> str:
    """Commit hash of the local Hugging Face snapshot, e.g. '167aab52…'. Never downloads."""
    from huggingface_hub import snapshot_download
    return os.path.basename(snapshot_download(repo_id, local_files_only=True))


def _few_shot_pool_sha256() -> str:
    """Hash of the GRADE GT pool get_few_shot_examples samples from; changes when GT changes."""
    rows = []
    for ds in load_acp_datasets() + load_icu_datasets():
        for row in ds.ground_truth_df[list(_POOL_COLUMNS)].astype(str).itertuples(index=False):
            rows.append([ds.doi, *row])
    return hashlib.sha256(_canonical({"pool": sorted(rows)}).encode("utf-8")).hexdigest()
