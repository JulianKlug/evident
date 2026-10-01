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
