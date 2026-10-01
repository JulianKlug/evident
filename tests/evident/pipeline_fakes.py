"""Fakes for pipeline and E2E tests: tmp manifest + PDFs, canned runner, fixed probe, small labelled set.

    World(tmp) ──deps()──▶ PipelineDeps(store=tmp DB, runner=FakeRunner, probe=FakeProbe, ...)
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Optional

from evident.domain import Category, GradingFamily, LlmCall, RawRecommendation, ThinkingMode
from evident.extraction import Artifacts, ExtractionMode, ExtractionOutput, ExtractorConfig, FewShotMode
from evident.pipeline import PipelineDeps
from evident.store import Store
from evident.validation import GtRow, LabelledGuideline
from tests.legacy_golden import FakeEncoder
from utils.doi import doi_to_filename

CONFIG = ExtractorConfig(mode=ExtractionMode.SINGLE_PASS, model="qwen3:14b", thinking=ThinkingMode.OFF,
                         prompt_version="v0", few_shot=FewShotMode.FEW_SHOT)
WORSE_CONFIG = replace(CONFIG, model="qwen3:8b")  # FakeRunner returns fewer recs for it

_HEADER = "doi,society,year,title,topic_id,supersedes_doi,grading_family,pdf_filename\n"

# doi → (society, year, topic, recommendation texts with grades)
GUIDELINES = {
    "10.1000/a": ("ACP", 2023, "topic-a", [
        ("ACP recommends against routine imaging in acute low back pain", "strong recommendation", "low"),
        ("ACP suggests exercise therapy for chronic knee osteoarthritis", "conditional recommendation", "moderate"),
        ("ACP recommends metformin as first line therapy in type two diabetes", "strong recommendation", "high"),
    ]),
    "10.1000/b": ("ESICM", 2024, "topic-b", [
        ("We suggest balanced crystalloids rather than saline for resuscitation", "conditional", "low"),
        ("We recommend against starches for volume replacement in sepsis", "strong", "high"),
        ("We suggest a restrictive fluid strategy after initial stabilisation", "conditional", "very low"),
    ]),
    "10.1000/c": ("ESICM", 2025, "topic-c", [
        ("We suggest diuretics to remove fluid in overloaded patients", "conditional", "low"),
    ]),
    "10.1000/d": ("ACP", 2025, "topic-d", [
        ("ACP suggests cognitive behavioural therapy for insomnia disorder", "conditional recommendation", "low"),
    ]),
}
LABELLED_DOIS = ("10.1000/a", "10.1000/b")


class FakeRunner:
    """Canned recs per DOI; `fail` lists DOIs that raise once each; WORSE_CONFIG drops the last 2 recs."""

    def __init__(self, fail: tuple[str, ...] = (), on_call=None):
        self.calls: list[str] = []
        self._fail = set(fail)
        self._on_call = on_call

    def __call__(self, cfg: ExtractorConfig, pdf_path: str, doi: str) -> ExtractionOutput:
        self.calls.append(doi)
        if self._on_call:
            self._on_call(len(self.calls))
        if doi in self._fail:
            self._fail.discard(doi)
            raise RuntimeError(f"ollama crashed on {doi}")

        rows = GUIDELINES[doi][3]
        if cfg == WORSE_CONFIG:
            rows = rows[:1]
        recs = [RawRecommendation(i, text, s, c, Category.GRADED, page=1, chunk_id=0)
                for i, (text, s, c) in enumerate(rows)]
        calls = [LlmCall(chunk_id=0, prompt_sha256="p", raw_text="r", eval_tokens=10)]
        return ExtractionOutput(recs=recs, calls=calls, n_pages=1, few_shot=[{"recommendation": "x"}])


class FakeProbe:
    def __init__(self):
        self.artifacts = Artifacts("digest", None, "rev", "pool")

    def __call__(self, cfg: ExtractorConfig) -> Artifacts:
        return self.artifacts


def labelled() -> list[LabelledGuideline]:
    return [
        LabelledGuideline(doi, GradingFamily.GRADE,
                          [GtRow(text, s, c, Category.GRADED) for text, s, c in GUIDELINES[doi][3]])
        for doi in LABELLED_DOIS
    ]


class World:
    """A tmp manifest, PDF dir and DB; `deps()` wires the fakes in."""

    def __init__(self, tmp_path, dois=("10.1000/a", "10.1000/b", "10.1000/c"), runner: Optional[FakeRunner] = None):
        self.root = str(tmp_path)
        self.manifest = os.path.join(self.root, "manifest.csv")
        self.pdf_dir = os.path.join(self.root, "pdfs")
        self.headline = os.path.join(self.root, "out", "headline.csv")
        os.makedirs(self.pdf_dir)
        self.store = Store.open(os.path.join(self.root, "db.sqlite"))
        self.runner = runner or FakeRunner()
        self.probe = FakeProbe()
        self.dois: list[str] = []
        self._rows: list[str] = []
        for doi in dois:
            self.add_row(doi)

    def add_row(self, doi: str, write_pdf: bool = True, supersedes: str = "", topic: str = "") -> None:
        society, year, default_topic, _ = GUIDELINES[doi]
        self.dois.append(doi)
        self._rows.append(f"{doi},{society},{year},Title {doi},{topic or default_topic},{supersedes},grade,\n")
        with open(self.manifest, "w") as f:
            f.write(_HEADER + "".join(self._rows))
        if write_pdf:
            self.write_pdf(doi)

    def write_pdf(self, doi: str, directory: Optional[str] = None) -> str:
        path = os.path.join(directory or self.pdf_dir, doi_to_filename(doi))
        with open(path, "wb") as f:
            f.write(b"%PDF " + doi.encode())  # distinct bytes: one PDF per DOI
        return path

    def deps(self, runner=None) -> PipelineDeps:
        return PipelineDeps(
            store=self.store, manifest_path=self.manifest, pdf_dir=self.pdf_dir,
            runner=runner or self.runner, probe=self.probe, labelled=labelled,
            similarity_model=FakeEncoder(), code_sha=lambda: "sha1", headline_path=self.headline,
        )
