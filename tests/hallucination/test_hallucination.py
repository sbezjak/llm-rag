"""Hallucinated-source tests.

The property: the system must never cite a chunk id that is not in the
index. A citation is the system's claim "this came from that source", so
a cited id with no matching chunk is a fabricated source, the citation
equivalent of a made-up fact.

This is checked against the RAW bracket contents, not `parse_citations`.
`parse_citations` filters out any id not in the context (by design, it
returns only usable citations), so a hallucinated id is invisible to it.
`parse_all_citations` keeps every bracketed id, which is what we compare
against the index. (See the docstrings on both in
llm_rag/generators/generate.py.)

Why we filter to chunk-id-SHAPED tokens. The citation regex matches any
`[...]`, so bracketed CODE in an answer (e.g. `pytest.main(["-m",
"x=y"])`) parses as citations too, a parser-fragility finding in its own
right (notes.md S5), and exactly the cost the inline-citation trade-off
in generate.py flagged. A hallucinated SOURCE is specifically a token
shaped like a chunk id (`source::section::n`) that is not in the index;
code literals like `"-m"` are not chunk-id-shaped and are not what this
gate is about. So the live test compares only `::`-shaped bracket tokens
against the index.

Why out-of-corpus questions, not in-corpus. Hallucinated sources surface
when the corpus does NOT contain the answer: in-corpus questions get a
real chunk and the model cites it correctly, so they almost never
fabricate an id. The out-of-corpus set is where the model is tempted to
invent a source (ooc-002 cites `how-to__plugin::registering-a-plugin::0`,
which does not exist), so that is what we run here.

Two layers:
  - test_detects_hallucinated_citation (mocked): proves the detection
    seam itself. A canned answer cites a bogus id; we assert
    parse_all_citations surfaces it while parse_citations hides it. This
    is the deterministic guarantee that the live test below can actually
    catch a hallucinated source. No Ollama, no Chroma.
  - test_no_hallucinated_source_live (ollama + chroma): runs real
    generation over real retrieval of out-of-corpus questions and asserts
    no chunk-id-shaped bracket token falls outside the index.
"""

from __future__ import annotations

import logging

import pytest

from llm_rag.generators import (
    generate_answer,
    parse_all_citations,
    parse_citations,
)
from llm_rag.providers.ollama import OllamaProvider
from llm_rag.retrievers.hybrid import hybrid_then_rerank

logger = logging.getLogger(__name__)


@pytest.mark.mocked
async def test_detects_hallucinated_citation():
    """parse_all_citations surfaces a fabricated id; parse_citations hides it."""
    context_id = "getting-started::first-test::0"
    bogus_id = "totally-made-up::section::9"
    # Model answer that cites both a real context id and a fabricated one.
    answer = f"Write a test function and assert on the result [{context_id}]. " \
             f"You can also enable telepathy mode [{bogus_id}]."
    known_ids = {context_id}

    raw = parse_all_citations(answer)
    filtered = parse_citations(answer, known_ids)

    logger.info("answer:\n%s", answer)
    logger.info("raw bracketed ids (parse_all_citations): %s", raw)
    logger.info("filtered ids (parse_citations): %s", filtered)

    # The seam the live test relies on: the fabricated id is visible in
    # the raw parse and absent from the filtered one.
    assert bogus_id in raw, "parse_all_citations must surface the fabricated id"
    assert bogus_id not in filtered, "parse_citations must drop the fabricated id"

    hallucinated = [cid for cid in raw if cid not in known_ids]
    assert hallucinated == [bogus_id]


def chunk_id_shaped(raw_ids: list[str]) -> list[str]:
    """Keep only bracket tokens shaped like a chunk id (`source::section::n`).

    Drops code literals the citation regex catches (e.g. `"-m"`) so the
    hallucinated-SOURCE gate is about fabricated chunk ids, not bracketed
    code. The shape test is deliberately loose: any token containing "::"
    is treated as an attempted source citation.
    """
    return [cid for cid in raw_ids if "::" in cid]


# Same disposition as the refusal suite's ooc-002 (see tests/refusal): the
# model stably (5/5, temperature 0) fabricates the source
# `how-to__plugin::registering-a-plugin::0` for the plugin-authoring question.
# strict=True so a future run that stops fabricating XPASS-fails and forces a
# re-read of the finding.
_XFAIL_HALLUCINATION = {
    "ooc-002": "S5 finding: model fabricates a non-existent source id "
    "(how-to__plugin::...) for an adjacent-absent question (5/5 runs)",
}


@pytest.fixture(scope="module")
def provider() -> OllamaProvider:
    return OllamaProvider(temperature=0.0, timeout=240.0)


@pytest.fixture
def ooc_entry(request, out_of_corpus_entries):
    return next(e for e in out_of_corpus_entries if e["id"] == request.param)


def _param_for(oid: str):
    marks = []
    if oid in _XFAIL_HALLUCINATION:
        marks.append(pytest.mark.xfail(reason=_XFAIL_HALLUCINATION[oid], strict=True))
    return pytest.param(oid, marks=marks)


def pytest_generate_tests(metafunc):
    if "ooc_entry" in metafunc.fixturenames:
        from llm_rag.dataset import load_yaml
        from tests.hallucination.conftest import OUT_OF_CORPUS_SET

        ids = [e["id"] for e in load_yaml(OUT_OF_CORPUS_SET)["entries"]]
        metafunc.parametrize("ooc_entry", [_param_for(o) for o in ids], indirect=True)


@pytest.mark.ollama
@pytest.mark.chroma
async def test_no_hallucinated_source_live(
    ooc_entry, provider, chroma_collection, bm25_index, index_ids, reranker
):
    """No chunk-id-shaped bracket token may fall outside the index.

    Run over out-of-corpus questions, where the model is tempted to invent
    a source. A `::`-shaped bracket token that is not an index id is a
    fabricated source: the citation equivalent of a made-up fact.
    """
    query = ooc_entry["query"]
    chunks = hybrid_then_rerank(
        query=query, k=5, collection=chroma_collection,
        bm25_index=bm25_index, reranker=reranker,
    )
    result = await generate_answer(query, chunks, provider)

    raw = parse_all_citations(result.answer)
    shaped = chunk_id_shaped(raw)
    hallucinated = [cid for cid in shaped if cid not in index_ids]

    logger.info("entry=%s query=%s", ooc_entry["id"], query)
    logger.info("answer:\n%s", result.answer)
    logger.info("raw bracketed ids: %s", raw)
    logger.info("chunk-id-shaped tokens: %s", shaped)
    logger.info("hallucinated (shaped, not in index): %s", hallucinated)

    assert not hallucinated, (
        f"{ooc_entry['id']}: cited source id(s) absent from the index: {hallucinated}"
    )
