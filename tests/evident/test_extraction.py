"""Tests for evident.extraction: config identity, artifact checks, run() over the legacy extractor."""

import dataclasses
import json
from enum import Enum

import pandas as pd
import pytest

import evident.extraction as ex
import extraction.extractor as extractor_mod
import extraction.self_consistency as sc_mod
from evident.domain import Category, ThinkingMode
from evident.extraction import (
    ArtifactMismatchError,
    Artifacts,
    ConsensusMode,
    ExtractionMode,
    ExtractorConfig,
    FewShotMode,
    Models,
    PdfMissingError,
    check_artifacts,
    version_for,
)
from extraction.llm_client import LLMResponse, ThinkMode
from extraction.pdf_loader import PDFPage
from tests.legacy_golden import FakeEncoder, expected_path, load_fixture

_CFG = ExtractorConfig(mode=ExtractionMode.SINGLE_PASS, model="qwen3:14b", thinking=ThinkingMode.OFF,
                       prompt_version="v0", few_shot=FewShotMode.FEW_SHOT)
_ARTIFACTS = Artifacts(model_digest="d", classifier_sha256=None, embedding_revision="r",
                       few_shot_pool_sha256="p")
_EXAMPLES = [{"recommendation": "Use X", "class": "Strong For", "LOE": "High"}]
_DOI = "10.7326/M22-2056"


def _changed(value):
    """A different value of the same type, for every config field type."""
    if isinstance(value, Enum):
        return next(m for m in type(value) if m != value)
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if value == "v0":
        return "v1"
    return value + "x"


class TestConfig:
    def test_json_round_trip(self):
        assert ExtractorConfig.from_json(_CFG.to_json()) == _CFG

    def test_canonical_form(self):
        data = json.loads(_CFG.to_json())
        shuffled = json.dumps(dict(reversed(list(data.items()))), indent=2)

        assert ExtractorConfig.from_json(shuffled).to_json() == _CFG.to_json()
        assert " " not in _CFG.to_json()
        assert data["mode"] == "single_pass" and data["thinking"] == "off"

    def test_unknown_field_and_prompt_refused(self):
        with pytest.raises(ValueError):
            ExtractorConfig.from_json(json.dumps({**json.loads(_CFG.to_json()), "normalize": True}))
        with pytest.raises(ValueError):
            dataclasses.replace(_CFG, prompt_version="v9")
        with pytest.raises(ValueError):
            dataclasses.replace(_CFG, embedding_model="other/model")


class TestVersion:
    @pytest.mark.parametrize("name", [f.name for f in dataclasses.fields(ExtractorConfig)
                                      if f.name != "embedding_model"])  # single allowed value
    def test_id_changes_with_each_config_field(self, name):
        other = dataclasses.replace(_CFG, **{name: _changed(getattr(_CFG, name))})
        assert version_for(other, _ARTIFACTS).id != version_for(_CFG, _ARTIFACTS).id

    @pytest.mark.parametrize("name", [f.name for f in dataclasses.fields(Artifacts)])
    def test_id_changes_with_each_artifact(self, name):
        other = dataclasses.replace(_ARTIFACTS, **{name: "changed"})
        assert version_for(_CFG, other).id != version_for(_CFG, _ARTIFACTS).id

    def test_version_fields(self):
        v = version_for(_CFG, _ARTIFACTS)
        assert (v.config_json, v.model_name, v.model_digest, v.embedding_revision) == (
            _CFG.to_json(), "qwen3:14b", "d", "r")

    @pytest.mark.parametrize("name", [f.name for f in dataclasses.fields(Artifacts)])
    def test_check_artifacts_names_the_field(self, name):
        version = version_for(_CFG, _ARTIFACTS)
        with pytest.raises(ArtifactMismatchError) as err:
            check_artifacts(version, dataclasses.replace(_ARTIFACTS, **{name: "changed"}))
        assert err.value.field == name

    def test_check_artifacts_ok(self):
        check_artifacts(version_for(_CFG, _ARTIFACTS), _ARTIFACTS)


class _Client:
    """Canned responses; records every kwarg so seed/think forwarding is visible."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.model = "mock"
        self.model_info = {"context_window": 4096, "has_thinking": False}
        self.has_thinking = False
        self.kwargs = []

    def generate(self, prompt, num_ctx=None, temperature=0, **kwargs):
        self.kwargs.append({"temperature": temperature, **kwargs})
        text = self._responses[len(self.kwargs) - 1]
        return LLMResponse(raw_text=text, model="mock", prompt_tokens=3, eval_tokens=5, total_duration_ms=7.0)


@pytest.fixture
def acp(monkeypatch, tmp_path):
    """ACP golden fixture wired in: pages, canned client, spied few-shot sampler, a real PDF path."""
    fx = load_fixture("acp")
    pages = [PDFPage(page_number=i + 1, text=t) for i, t in enumerate(fx["pages"])]
    monkeypatch.setattr(extractor_mod, "load_pdf_pages", lambda source: pages)
    monkeypatch.setattr(sc_mod, "load_pdf_pages", lambda source: pages)

    few_shot_calls = []

    def fake_examples(scheme, n_examples, exclude_doi=None, seed=None):
        few_shot_calls.append({"n": n_examples, "exclude_doi": exclude_doi, "seed": seed})
        return _EXAMPLES
    monkeypatch.setattr(ex, "get_few_shot_examples", fake_examples)

    responses = fx["responses"] + [r for sample in fx["sc_responses"] for r in sample]
    client = _Client(responses)
    monkeypatch.setattr(ex, "OllamaClient", lambda model: client)

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    cfg = dataclasses.replace(_CFG, pages_per_chunk=fx["pages_per_chunk"])
    return {"cfg": cfg, "pdf": str(pdf), "client": client, "few_shot_calls": few_shot_calls}


class TestRun:
    def test_missing_pdf_never_reaches_legacy(self, monkeypatch, tmp_path):
        def spy(source):
            raise AssertionError("load_pdf_pages called")
        monkeypatch.setattr(extractor_mod, "load_pdf_pages", spy)

        with pytest.raises(PdfMissingError):
            ex.run(_CFG, str(tmp_path / "missing.pdf"), _DOI, Models(similarity=None))

    def test_single_pass_raw_output(self, acp):
        out = ex.run(acp["cfg"], acp["pdf"], _DOI, Models(similarity=None))

        # Same rows as the un-normalized legacy golden: raw class/LOE kept
        expected = pd.read_csv(expected_path("acp_fewshot"), dtype=str, keep_default_na=False)
        assert [(r.text, r.raw_strength, r.raw_certainty) for r in out.recs] == list(
            expected.itertuples(index=False, name=None))
        assert [r.ordinal for r in out.recs] == list(range(len(out.recs)))
        assert all(r.raw_category == Category.GRADED and r.page and r.chunk_id is not None for r in out.recs)

    def test_calls_and_few_shot_recorded(self, acp):
        out = ex.run(acp["cfg"], acp["pdf"], _DOI, Models(similarity=None))

        assert len(out.calls) == len(acp["client"].kwargs)
        assert [c.chunk_id for c in out.calls] == list(range(len(out.calls)))
        assert out.calls[0].eval_tokens == 5 and out.calls[0].duration_ms == 7.0
        assert out.few_shot == _EXAMPLES
        assert acp["few_shot_calls"] == [{"n": 3, "exclude_doi": _DOI, "seed": 0}]

    def test_single_pass_forwards_seed_and_think(self, acp):
        ex.run(acp["cfg"], acp["pdf"], _DOI, Models(similarity=None))
        assert acp["client"].kwargs[0] == {"temperature": 0, "seed": 0, "think": ThinkMode.OFF}

    def test_zero_shot_skips_examples(self, acp):
        cfg = dataclasses.replace(acp["cfg"], few_shot=FewShotMode.ZERO_SHOT)
        out = ex.run(cfg, acp["pdf"], _DOI, Models(similarity=None))
        assert out.few_shot == [] and acp["few_shot_calls"] == []

    def test_sc_dispatch(self, acp, monkeypatch):
        seen = {}

        def fake_sc(source, **kwargs):
            seen.update(kwargs)
            return sc_mod.ExtractionResult(pd.DataFrame(columns=["recommendation"]), 0, 0, 0, 0)
        monkeypatch.setattr(ex, "self_consistency_extract", fake_sc)
        cfg = dataclasses.replace(acp["cfg"], mode=ExtractionMode.SELF_CONSISTENCY, seed=4,
                                  thinking=ThinkingMode.ON, sc_consensus=ConsensusMode.FIXED)
        encoder = FakeEncoder()

        ex.run(cfg, acp["pdf"], _DOI, Models(similarity=encoder))

        assert (seen["seed"], seen["think"], seen["adaptive_threshold"], seen["normalize"]) == (
            4, ThinkMode.ON, False, False)
        assert seen["similarity_model"] is encoder and seen["n_samples"] == 3
