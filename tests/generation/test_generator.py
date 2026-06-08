"""Generator wiring tests (mocked, no Ollama).

These assert the *mechanics* of generation: the prompt pins each chunk
to its id and demands inline citation, and citations are parsed back out
correctly. Answer quality is a separate concern, scored in
test_generation_quality.py against a real model.
"""

from __future__ import annotations

import pytest

from llm_rag.generators import build_prompt, generate_answer, parse_citations
from llm_rag.retrievers.vector import RetrievedChunk
from tests.generation.conftest import FakeProvider, chunks_for


def _chunk(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(id=cid, text=text, score=1.0, source="src", section="sec")


@pytest.mark.mocked
def test_prompt_includes_every_chunk_id_and_citation_instruction():
    chunks = [_chunk("a::x::0", "alpha text"), _chunk("b::y::1", "beta text")]
    prompt = build_prompt("a question", chunks)

    assert "[a::x::0]" in prompt
    assert "[b::y::1]" in prompt
    assert "alpha text" in prompt and "beta text" in prompt
    assert "square brackets" in prompt
    assert "a question" in prompt


@pytest.mark.mocked
def test_parse_citations_keeps_known_ids_in_order_deduped():
    known = {"a::x::0", "b::y::1"}
    answer = "First point [a::x::0]. Second [b::y::1]. Repeat [a::x::0]."
    assert parse_citations(answer, known) == ["a::x::0", "b::y::1"]


@pytest.mark.mocked
def test_parse_citations_handles_multiple_ids_in_one_bracket():
    known = {"a::x::0", "b::y::1"}
    answer = "Combined claim [a::x::0, b::y::1]."
    assert parse_citations(answer, known) == ["a::x::0", "b::y::1"]


@pytest.mark.mocked
def test_parse_citations_drops_ids_not_in_context():
    # A cited id that was never in the context is dropped here; the S5
    # hallucinated-source tests are what assert on that separately.
    known = {"a::x::0"}
    answer = "Grounded [a::x::0] but also invented [made::up::9]."
    assert parse_citations(answer, known) == ["a::x::0"]


@pytest.mark.mocked
async def test_generate_answer_returns_answer_and_cited_ids():
    chunks = [_chunk("a::x::0", "alpha text")]
    provider = FakeProvider(answer="Do the thing [a::x::0].")

    result = await generate_answer("how?", chunks, provider)

    assert result.answer == "Do the thing [a::x::0]."
    assert result.cited_chunk_ids == ["a::x::0"]
    # The provider really saw the citation-forcing prompt.
    assert "[a::x::0]" in provider.last_prompt
    # NamedTuple still unpacks as the original (answer, cited) contract.
    answer, cited = result
    assert answer == result.answer and cited == result.cited_chunk_ids


@pytest.mark.mocked
async def test_generate_answer_over_real_known_good_context(generation_entries):
    """End-to-end shape check on a real corpus chunk, fake model.

    Uses the first generation-set row's known-good context so the prompt
    is built over real chunk text, but a FakeProvider stands in for the
    LLM, so this stays fast and deterministic.
    """
    entry = generation_entries[0]
    chunks = chunks_for(entry)
    cid = chunks[0].id
    provider = FakeProvider(answer=f"Here is how [{cid}].")

    result = await generate_answer(entry["query"], chunks, provider)

    assert result.cited_chunk_ids == [cid]
    assert chunks[0].text in provider.last_prompt
