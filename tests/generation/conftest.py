"""Shared fixtures for generation tests.

Generation tests inject hand-blessed known-good context as a fixture and
never call the retrieval layer, so a retrieval bug cannot make a
generation test fail. The context comes from data/generation_set.yaml,
where each row's `known_good_context` was pulled verbatim from the index
at scaffold time.

The scorers are session-scoped: SemanticScorer loads a sentence
transformer (seconds, plus a one-time download) and we don't want to pay
that per test.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from llm_rag.dataset import load_yaml
from llm_rag.providers.base import Provider
from llm_rag.retrievers.vector import RetrievedChunk

ROOT = Path(__file__).resolve().parents[2]
GENERATION_SET = ROOT / "data" / "generation_set.yaml"


@pytest.fixture(scope="session")
def generation_entries() -> list[dict]:
    return load_yaml(GENERATION_SET)["entries"]


def chunks_for(entry: dict) -> list[RetrievedChunk]:
    """Build the known-good context for an entry as RetrievedChunks.

    Each generation row currently carries a single known-good chunk: the
    full text under `known_good_context`, tied to the first id in
    `context_chunk_ids` for provenance. The retriever's `score`/`source`
    /`section` fields are irrelevant to generation, so they get filler
    values; only `id` and `text` matter here.
    """
    cid = entry["context_chunk_ids"][0]
    source = cid.split("::", 1)[0]
    return [
        RetrievedChunk(
            id=cid,
            text=entry["known_good_context"],
            score=1.0,
            source=source,
            section=cid.split("::")[1] if "::" in cid else "",
        )
    ]


class FakeProvider(Provider):
    """Provider that returns a canned answer, recording the prompt it saw.

    Lets generation tests run without Ollama: the test controls exactly
    what the "model" says (including which chunk ids it cites), so prompt
    construction and citation parsing can be asserted deterministically.
    """

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.last_prompt: str | None = None

    async def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.answer


def fake_judge_fn(payload: str) -> Callable[[str], Awaitable[str]]:
    """Build an async judge_fn that always returns `payload`."""

    async def _fn(prompt: str) -> str:
        return payload

    return _fn
