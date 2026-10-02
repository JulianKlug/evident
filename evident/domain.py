"""Domain enums and records shared by all evident layers. No logic.

All enums are str-valued so they serialize to the DB as-is.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class GradingFamily(str, Enum):
    GRADE = "grade"  # M6 adds ABCD_123, ESC_ERS, ACC_AHA


class Strength(str, Enum):
    STRONG = "strong"
    WEAK = "weak"


class Direction(str, Enum):
    FOR = "for"
    AGAINST = "against"


class Certainty(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"


class Category(str, Enum):
    GRADED = "graded"
    BEST_PRACTICE = "best_practice"
    NO_RECOMMENDATION = "no_recommendation"


class AxisStatus(str, Enum):
    MAPPED = "mapped"
    UNGRADED = "ungraded"
    UNMAPPED = "unmapped"


class RunStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    FAILED_EMPTY = "failed_empty"


class SnapshotState(str, Enum):
    BUILDING = "building"
    COMPLETE = "complete"
    PUBLISHED = "published"
    REJECTED = "rejected"  # terminal: a candidate never to be published, with a reason


class ThinkingMode(str, Enum):
    OFF = "off"
    ON = "on"


class MemberOrigin(str, Enum):
    INITIAL = "initial"            # in the manifest when the snapshot was created
    RESCAN = "rescan"              # added to the manifest while the snapshot was building
    POST_PUBLISH = "post_publish"  # added to a published snapshot by `add`


class GateResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NO_BASELINE = "no_baseline"


@dataclass(frozen=True)
class Guideline:
    doi: str
    society: str
    year: int
    title: str
    topic_id: str
    grading_family: GradingFamily
    supersedes_doi: Optional[str] = None
    pdf_path: Optional[str] = None
    pdf_sha256: Optional[str] = None
    added_at: Optional[str] = None  # set by the store on insert
    id: Optional[int] = None


@dataclass(frozen=True)
class ExtractorVersion:
    id: str
    config_json: str
    model_name: str
    model_digest: str
    classifier_sha256: Optional[str] = None
    embedding_revision: Optional[str] = None
    created_at: Optional[str] = None


@dataclass(frozen=True)
class Snapshot:
    id: int
    extractor_version_id: str
    state: SnapshotState
    created_at: str
    completed_at: Optional[str] = None
    published_at: Optional[str] = None
    accept_reason: Optional[str] = None
    rejected_at: Optional[str] = None
    reject_reason: Optional[str] = None


@dataclass(frozen=True)
class ExtractionRun:
    id: int
    snapshot_id: int
    guideline_id: int
    status: RunStatus
    started_at: str
    thinking: ThinkingMode
    finished_at: Optional[str] = None
    duration_s: Optional[float] = None
    n_pages: Optional[int] = None
    n_llm_calls: Optional[int] = None
    few_shot_json: Optional[str] = None
    error: Optional[str] = None
    code_sha: Optional[str] = None  # git HEAD at run time; not part of the version


@dataclass(frozen=True)
class SnapshotMember:
    guideline_id: int
    excluded_reason: Optional[str]
    origin: MemberOrigin


@dataclass(frozen=True)
class Validation:
    id: int
    snapshot_id: int
    baseline_snapshot_id: Optional[int]
    gate: GateResult
    report_json: str
    created_at: str


@dataclass(frozen=True)
class RawRecommendation:
    ordinal: int
    text: str
    raw_strength: str  # exactly as extracted, before any normalization
    raw_certainty: str
    raw_category: Category
    page: Optional[int] = None  # 1-based; None only for legacy rows
    chunk_id: Optional[int] = None


@dataclass(frozen=True)
class LlmCall:
    chunk_id: int
    prompt_sha256: str
    raw_text: str
    sample_idx: int = 0
    seed: Optional[int] = None
    prompt_tokens: Optional[int] = None
    eval_tokens: Optional[int] = None
    duration_ms: Optional[float] = None


@dataclass(frozen=True)
class HarmonizedGrade:
    category: Category
    strength: Optional[Strength]
    direction: Optional[Direction]
    certainty: Optional[Certainty]
    strength_status: AxisStatus
    certainty_status: AxisStatus
