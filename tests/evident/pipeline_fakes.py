"""Fakes for pipeline and E2E tests: tmp manifest + PDFs, canned runner, fixed probe, small labelled set.

    World(tmp) ──deps()──▶ PipelineDeps(store=tmp DB, runner=FakeRunner, probe=FakeProbe, ...)
    LiveWorld(tmp) ──live()──▶ + candidates, search window, verdict ledger, release archive (M5)
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
from dataclasses import replace
from typing import Optional

import pandas as pd

from evident.dedup import Verdict, merge_rows, read_verdict_rows, write_adjudications
from evident.domain import Category, GradingFamily, LlmCall, RawRecommendation, ThinkingMode
from evident.extraction import Artifacts, ExtractionMode, ExtractionOutput, ExtractorConfig, FewShotMode
from evident.corpus import file_sha256
from evident.pipeline import PipelineDeps, duplicate_candidates
from evident.store import Store
from evident.validation import GtRow, LabelledGuideline
from tests.legacy_golden import FakeEncoder
from utils.doi import doi_to_filename

CONFIG = ExtractorConfig(mode=ExtractionMode.SINGLE_PASS, model="qwen3:14b", thinking=ThinkingMode.OFF,
                         prompt_version="v0", few_shot=FewShotMode.FEW_SHOT)
WORSE_CONFIG = replace(CONFIG, model="qwen3:8b")  # FakeRunner returns fewer recs for it
FEW_SHOT_EXAMPLE = {"recommendation": "We suggest early mobilisation", "class": "Weak For", "LOE": "Low"}

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
    "10.1000/e": ("SCCM", 2025, "topic-e", [  # held out in package tests
        ("We recommend active cooling for patients with exertional heat stroke", "strong", "moderate"),
        ("We suggest against dantrolene for heat stroke", "conditional", "very low"),
    ]),
    "10.1000/f": ("ESICM", 2026, "topic-f", [  # M5: a late guideline whose summary list repeats a statement
        ("We suggest prone positioning in severe ARDS", "conditional", "moderate"),
        ("We recommend lung protective ventilation in ARDS", "strong", "high"),
        ("We suggest prone positioning in severe ARDS (conditional recommendation)", "conditional", "moderate"),
    ]),
}
LABELLED_DOIS = ("10.1000/a", "10.1000/b")

# Re-downloaded PDF bytes (M5 rebind-pdf), appended to a fake PDF; fake_prompt_hashes reads them
META_ONLY = b" %meta re-downloaded"              # bytes change, extracted text does not
TEXT_CHANGE = b" TEXT downloaded on 2026-10-25"  # a visible stamp: the prompt changes
EXTRA_PAGE = b" PAGES"                           # one more chunk


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
        return ExtractionOutput(recs=recs, calls=calls, n_pages=1, few_shot=[FEW_SHOT_EXAMPLE])


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
        self.publish_dir = os.path.join(self.root, "out")
        self.headline = os.path.join(self.publish_dir, "tables", "headline.csv")
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
            similarity_model=FakeEncoder(), code_sha=lambda: "sha1", publish_dir=self.publish_dir,
        )


_CANDIDATE_HEADER = ["doi", "society", "year", "title", "journal", "pmid", "source", "decision", "reason",
                     "duplicate_of"]
def fake_prompt_hashes(cfg, pdf_path, doi) -> dict[int, str]:
    """FakeRunner's single chunk has prompt hash "p"; marker bytes stand for a text or page change."""
    with open(pdf_path, "rb") as f:
        body = f.read()
    if TEXT_CHANGE.strip() in body:
        return {0: "q"}
    if EXTRA_PAGE.strip() in body:
        return {0: "p", 1: "p"}
    return {0: "p"}


class LiveWorld(World):
    """World plus the live inputs the CLI passes: candidates, search window, verdict ledger, release archive."""

    def __init__(self, tmp_path, runner=None):
        super().__init__(tmp_path, runner=runner)
        self.candidates = os.path.join(self.root, "candidates.csv")
        self.search = os.path.join(self.root, "search.json")
        self.ledger = os.path.join(self.root, "live_verdicts.csv")
        self.archive = os.path.join(self.root, "releases")
        self.rebind_log = os.path.join(self.root, "pdf_rebinds.csv")
        with open(self.search, "w") as f:
            json.dump({"start": "2016-01-01", "cutoff": "2026-10-04"}, f)
        write_adjudications(self.ledger, [])
        self.write_candidates()

    def add_guideline(self, doi: str, **row) -> None:
        """A curated guideline: manifest row and included candidate, PDF in the PDF dir."""
        self.add_row(doi, **row)
        self.write_candidates()

    def write_candidates(self, extra: tuple[str, ...] = ()) -> None:
        with open(self.candidates, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(_CANDIDATE_HEADER)
            for doi in [*self.dois, *extra]:
                society, year = GUIDELINES[doi][:2]
                w.writerow([doi, society, year, f"Title {doi}", "J", "", "pubmed", "include", "", ""])

    def live(self, runner=None):
        return dataclasses.replace(
            self.deps(runner), candidates_path=self.candidates, search_path=self.search,
            dedup_verdicts_path=self.ledger, release_archive_dir=self.archive, prompt_hashes=fake_prompt_hashes,
            rebind_log_path=self.rebind_log)

    def judge(self, snapshot_id: int, duplicates=()) -> None:
        """Both judges agree: the listed pairs are duplicates, every other unjudged screened pair distinct."""
        pairs = duplicate_candidates(self.store, snapshot_id, self.manifest, FakeEncoder())
        new = [(p, Verdict.DUPLICATE if (p.doi, p.ordinal_a, p.ordinal_b) in duplicates else Verdict.DISTINCT,
                "reason", "m1+m2", "d1+d2") for p in pairs]
        write_adjudications(self.ledger, merge_rows(read_verdict_rows(self.ledger), new))

    def output_hashes(self) -> dict[str, str]:
        out = {}
        for root, _, files in os.walk(self.publish_dir):
            for name in files:
                path = os.path.join(root, name)
                out[os.path.relpath(path, self.publish_dir)] = file_sha256(path)
        return out

    def pdf(self, doi: str) -> str:
        return os.path.join(self.pdf_dir, doi_to_filename(doi))

    def append_to_pdf(self, doi: str, marker: bytes, directory: str = None) -> str:
        """The PDF re-downloaded with `marker` in it; in the PDF dir unless another directory is given."""
        with open(self.pdf(doi), "rb") as f:
            body = f.read()
        path = os.path.join(directory, doi_to_filename(doi)) if directory else self.pdf(doi)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(body + marker)
        return path


def write_heldout_xlsx(directory: str, doi: str, rows: list[tuple], sheet: str = "labels") -> str:
    """A held-out label file as annotators fill it: rows of (recommendation, class, LOE, page); Key/Title/DOI once."""
    os.makedirs(directory, exist_ok=True)
    records = [{"Key": doi if i == 0 else None, "Title": f"Title {doi}" if i == 0 else None,
                "DOI": doi if i == 0 else None, "recommendation": text, "class": cls, "LOE": loe, "page": page}
               for i, (text, cls, loe, page) in enumerate(rows)]
    if not records:
        records = [{"Key": doi, "Title": f"Title {doi}", "DOI": doi, "recommendation": None, "class": None,
                    "LOE": None, "page": None}]
    path = os.path.join(directory, doi.replace("/", "_") + ".xlsx")
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(records).to_excel(writer, sheet_name=sheet, index=False)
        pd.DataFrame({"rule": ["Label blind."]}).to_excel(writer, sheet_name="rules", index=False)
    return path
