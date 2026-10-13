"""Tests for extraction.llm_client option handling (ollama.Client patched)."""

import pytest

import extraction.llm_client as llm_mod
from extraction.llm_client import OllamaClient

_SCHEMA = {"type": "object"}


class _FakeOllama:
    def __init__(self, **kwargs):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return {"response": "x"}

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": "x"}}


@pytest.fixture
def make_client(monkeypatch):
    monkeypatch.setattr(llm_mod.ollama, "Client", _FakeOllama)
    return OllamaClient


@pytest.mark.parametrize("model", ["qwen3:14b", "qwen3-14b-ft"])  # generate and chat paths
def test_options_without_seed_unchanged(make_client, model):
    client = make_client(model=model)
    client.generate("p --- Guideline Text --- t", temperature=0.3)
    assert client._client.calls[0]["options"] == {"temperature": 0.3}


@pytest.mark.parametrize("model", ["qwen3:14b", "qwen3-14b-ft"])
def test_options_with_seed(make_client, model):
    client = make_client(model=model)
    client.generate("p --- Guideline Text --- t", temperature=0.3, seed=7)
    assert client._client.calls[0]["options"] == {"temperature": 0.3, "seed": 7}


def test_generate_json_seed(make_client):
    client = make_client(model="qwen3:14b")
    client.generate_json("p", schema=_SCHEMA)
    client.generate_json("p", schema=_SCHEMA, seed=3)
    assert [c["options"] for c in client._client.calls] == [{"temperature": 0}, {"temperature": 0, "seed": 3}]


@pytest.mark.parametrize("model", ["qwen3:14b", "qwen3-14b-ft"])
@pytest.mark.parametrize("think, expected", [
    (llm_mod.ThinkMode.DEFAULT, "absent"), (llm_mod.ThinkMode.ON, True), (llm_mod.ThinkMode.OFF, False),
])
def test_think_kwarg_only_when_set(make_client, model, think, expected):
    client = make_client(model=model)
    client.generate("p --- Guideline Text --- t", think=think)
    client.generate_json("p", schema=_SCHEMA, think=think)
    assert [c.get("think", "absent") for c in client._client.calls] == [expected, expected]


class _Entry:
    def __init__(self, model, digest):
        self.model = model
        self.digest = digest


class _ListResponse:
    models = [_Entry("qwen3:8b", "sha-8b"), _Entry("qwen3:14b", "sha-14b")]


def test_model_digest(make_client, monkeypatch):
    client = make_client(model="qwen3:14b")
    monkeypatch.setattr(client._client, "list", lambda: _ListResponse(), raising=False)
    assert client.model_digest() == "sha-14b"


def test_model_digest_missing_model(make_client, monkeypatch):
    client = make_client(model="llama3.2:latest")
    monkeypatch.setattr(client._client, "list", lambda: _ListResponse(), raising=False)
    with pytest.raises(llm_mod.ModelNotFoundError, match="llama3.2"):
        client.model_digest()
