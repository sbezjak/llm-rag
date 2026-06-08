from __future__ import annotations

import logging

import httpx

from .base import CompletionResult, Provider

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Raised when the Ollama backend returns an error or unexpected payload."""


class OllamaProvider(Provider):
    """Ollama `/api/generate` adapter.

    Shape ported from llm-eval-harness. Streaming is disabled
    (`stream=False`) so a single JSON response carries the full
    completion under the `response` key. Sampling params go through
    Ollama's `options` object; `temperature` defaults to 0.2 (low, for
    reproducibility), but the LLM-judge scorer constructs this provider
    with `temperature=0.0` so its grades are deterministic.

    Unlike the eval-harness original (which logs at DEBUG without
    content), this adapter logs the full prompt and response at INFO so
    the always-on pytest-html report shows exactly what the model saw
    and said. That visibility is the point of a RAG test suite, so it is
    not optional here.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature

    async def generate(self, prompt: str) -> str:
        """Return just the completion text. Delegates to `complete`."""
        return (await self.complete(prompt)).text

    async def complete(self, prompt: str) -> CompletionResult:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        logger.info("prompt (%s): %s", self.model, prompt)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                raise OllamaError(f"Ollama request failed: {e}") from e

        data = resp.json()
        if "response" not in data:
            raise OllamaError(f"Ollama response missing 'response' key: {data!r}")
        response = data["response"]
        logger.info("response (%s): %s", self.model, response)

        # Token / timing fields are absent from minimal mocks, so read
        # them defensively: a missing field means "unknown", not zero.
        total_ns = data.get("total_duration")
        return CompletionResult(
            text=response,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            total_duration_s=(total_ns / 1e9) if total_ns is not None else None,
        )
