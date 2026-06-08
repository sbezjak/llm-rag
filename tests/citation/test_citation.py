"""Citation-correctness tests (requires live Ollama).

The question here is narrow and different from generation quality: when
the model DOES cite, is the cited chunk the one it was actually given?
We feed each generation row its single hand-blessed known-good chunk
(retrieval bypassed, same as the generation suite) and split the claim
in two:

  1. Correctness (hard-asserted). Every id the model cites must be the
     context chunk it was handed. The model must never cite a chunk that
     was not in its context. With a single-chunk context this is a clean
     set assertion: cited ids subset of {context id}. A violation here
     is a real bug, the model invented a source label.

  2. Compliance (xfail, strict=False, on the known non-citers). Did the
     model cite at all? notes.md finding 6: citation propensity is
     per-row and per-session, gen-002 and gen-007 are terse "do X"
     answers that reliably omit the citation within a session but can
     flip across sessions. We mark those rows xfail(strict=False): a
     non-strict xfail documents the expected miss WITHOUT turning a
     good run (the model does cite, XPASS) into a suite failure. That is
     exactly why strict=False is the right tool here and was the wrong
     tool inside the bundled quality test, which strict-xfails on the
     semantic gate, a stable disagreement, not a flip.

Why this is not redundant with the citation line logged in
test_generation_quality. That logs compliance for inspection but gates
nothing. This file gates CORRECTNESS (cited == handed) hard, and treats
compliance as a documented expectation per row. The deterministic
prompt-build + parse contract still lives in the mocked
test_generator.py; this is the behavioural layer on top.
"""

from __future__ import annotations

import logging

import pytest

from llm_rag.generators import generate_answer
from llm_rag.providers.ollama import OllamaProvider
from tests.generation.conftest import GENERATION_SET, chunks_for

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.ollama

# Rows whose answers reliably omit the forced citation within a session
# (notes.md finding 6): terse single-sentence "do X" answers. strict=False
# so a cross-session flip to citing does not hard-fail the suite.
_XFAIL_COMPLIANCE = {
    "gen-002": "finding 6: terse answer, reliably omits citation within a session",
    "gen-007": "finding 6: terse answer, reliably omits citation within a session",
}


@pytest.fixture(scope="module")
def provider() -> OllamaProvider:
    return OllamaProvider(temperature=0.0)


@pytest.fixture
def gen_entry(request):
    # Load the generation set directly: its `generation_entries` fixture
    # lives in tests/generation/conftest.py and is not visible here.
    from llm_rag.dataset import load_yaml

    entries = load_yaml(GENERATION_SET)["entries"]
    return next(e for e in entries if e["id"] == request.param)


def _param_for(gid: str):
    marks = []
    if gid in _XFAIL_COMPLIANCE:
        marks.append(pytest.mark.xfail(reason=_XFAIL_COMPLIANCE[gid], strict=False))
    return pytest.param(gid, marks=marks)


def pytest_generate_tests(metafunc):
    if "gen_entry" in metafunc.fixturenames:
        from llm_rag.dataset import load_yaml

        entries = load_yaml(GENERATION_SET)["entries"]
        ids = [e["id"] for e in entries]
        metafunc.parametrize("gen_entry", [_param_for(g) for g in ids], indirect=True)


async def test_citation_correctness(gen_entry, provider):
    if gen_entry["expected_answer"] == "TODO":
        pytest.skip(f"{gen_entry['id']}: expected_answer not filled in yet")

    chunks = chunks_for(gen_entry)
    context_id = chunks[0].id
    query = gen_entry["query"]

    result = await generate_answer(query, chunks, provider)

    logger.info("entry=%s", gen_entry["id"])
    logger.info("query: %s", query)
    logger.info("context chunk id (the only valid citation): %s", context_id)
    logger.info("answer:\n%s", result.answer)
    logger.info("cited chunk ids: %s", result.cited_chunk_ids)

    # 1. Correctness: the model must not cite anything other than the
    #    single chunk it was handed. parse_citations already drops ids
    #    outside the context, so any survivor must be context_id; this
    #    asserts the parsed set is exactly within {context_id}.
    stray = [cid for cid in result.cited_chunk_ids if cid != context_id]
    assert not stray, (
        f"{gen_entry['id']}: cited a chunk not in context: {stray} "
        f"(only valid citation is {context_id})"
    )

    # 2. Compliance: did it cite at all? xfail(strict=False) on the rows
    #    that reliably omit it (see _XFAIL_COMPLIANCE).
    assert context_id in result.cited_chunk_ids, (
        f"{gen_entry['id']}: model did not cite its context chunk {context_id}"
    )
