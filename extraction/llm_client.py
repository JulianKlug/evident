"""Ollama LLM client wrapper for recommendation extraction."""

from __future__ import annotations

import time
from dataclasses import dataclass

import ollama


AVAILABLE_MODELS = {
    "deepseek-r1:32b": {"context_window": 16384, "has_thinking": True},
    "qwen3:8b": {"context_window": 16384, "has_thinking": False},
    "llama3.2:latest": {"context_window": 8192, "has_thinking": False},
    "qwen3:14b": {"context_window": 32768, "has_thinking": False},
    "gemma3:27b": {"context_window": 32768, "has_thinking": False},
    "mistral-small3.2:24b": {"context_window": 32768, "has_thinking": False},
    "qwen3:30b-a3b": {"context_window": 32768, "has_thinking": False},
}


@dataclass
class LLMResponse:
    """Response from an LLM generation call."""
    raw_text: str
    model: str
    prompt_tokens: int
    eval_tokens: int
    total_duration_ms: float


class OllamaClient:
    """Wrapper around the Ollama API for generating text."""

    def __init__(self, model: str = "deepseek-r1:32b", host: str | None = None):
        self.model = model
        self.model_info = AVAILABLE_MODELS.get(model, {"context_window": 4096, "has_thinking": False})
        kwargs = {}
        if host:
            kwargs["host"] = host
        self._client = ollama.Client(**kwargs)

    @property
    def has_thinking(self) -> bool:
        return self.model_info.get("has_thinking", False)

    def generate(self, prompt: str, num_ctx: int | None = None, temperature: float = 0) -> LLMResponse:
        """Generate a completion from the model.

        Args:
            prompt: The prompt text.
            num_ctx: Context window size. Only passed to Ollama when explicitly set,
                     otherwise uses the model's default context window.
            temperature: Sampling temperature (0 = deterministic).
        """
        options: dict = {"temperature": temperature}
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        start = time.time()
        response = self._client.generate(
            model=self.model,
            prompt=prompt,
            options=options,
        )
        elapsed_ms = (time.time() - start) * 1000

        return LLMResponse(
            raw_text=response.get("response", ""),
            model=self.model,
            prompt_tokens=response.get("prompt_eval_count", 0),
            eval_tokens=response.get("eval_count", 0),
            total_duration_ms=elapsed_ms,
        )

    def generate_json(
        self, prompt: str, schema: dict, num_ctx: int | None = None,
    ) -> LLMResponse:
        """Generate a JSON-structured completion using Ollama's format parameter.

        Args:
            prompt: The prompt text.
            schema: JSON schema for the expected output format.
            num_ctx: Context window size. Only passed to Ollama when explicitly set.
        """
        options: dict = {"temperature": 0}
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        start = time.time()
        response = self._client.generate(
            model=self.model,
            prompt=prompt,
            format=schema,
            options=options,
        )
        elapsed_ms = (time.time() - start) * 1000

        return LLMResponse(
            raw_text=response.get("response", ""),
            model=self.model,
            prompt_tokens=response.get("prompt_eval_count", 0),
            eval_tokens=response.get("eval_count", 0),
            total_duration_ms=elapsed_ms,
        )
