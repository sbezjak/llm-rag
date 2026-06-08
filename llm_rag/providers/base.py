from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class CompletionResult:
    """A completion plus whatever metadata the backend reported.

    `text` is the answer string (same value `generate` returns). The
    token / timing fields are Optional because not every backend (or
    every mock) reports them: a respx mock that returns only
    `{"response": ...}` yields a result with `None` token counts, which
    the observability layer records as "unknown" rather than zero.

    Token counts come straight from Ollama's `prompt_eval_count` /
    `eval_count`; `total_duration_s` is Ollama's `total_duration`
    (nanoseconds) converted to seconds. They are surfaced here, at the
    provider boundary, because that is the only place HTTP happens, the
    day-1 baseline reads them through `complete`.
    """

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_duration_s: float | None = None


class Provider(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str: ...

    async def complete(self, prompt: str) -> CompletionResult:
        """Generate and return text plus backend metadata.

        Default implementation wraps `generate` with no token metadata,
        so providers and test fakes that only implement `generate` keep
        working unchanged. Backends that report token usage (Ollama)
        override this and have `generate` delegate to it.
        """
        return CompletionResult(text=await self.generate(prompt))
