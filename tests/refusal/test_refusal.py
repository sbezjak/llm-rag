"""Refusal tests (requires live Ollama + a built Chroma index).

For each out-of-corpus question we run the REAL pipeline, retrieve and
rerank from the index, then generate, and assert the model declines
rather than answering from whatever near-miss chunk retrieval surfaced.
This is the honest-failure property of the RAG system: no answer in the
corpus, so the right output is "I don't know based on the provided
context.", not a confident answer built from an irrelevant chunk.

Why the full pipeline and not injected context (unlike generation
tests). The temptation to hallucinate comes precisely from the chunk
retrieval hands the model for an out-of-corpus query, a real but
irrelevant near-miss. Injecting empty or hand-picked context would test
a situation that never happens in production. So refusal is the one
category that deliberately couples retrieval and generation.

What counts as a refusal (lenient on wording, strict on substance). The
prompt mandates the exact string "I don't know based on the provided
context." We accept any clear declination (the model paraphrases on some
runs), but we reject a refusal that is followed by a substantive answer,
and we reject any answer that cites a chunk, the prompt says to cite
nothing when refusing, so a citation means it tried to answer.
"""

from __future__ import annotations

import logging

import pytest

from llm_rag.generators import build_prompt, generate_answer, parse_all_citations
from llm_rag.providers.ollama import OllamaProvider
from llm_rag.retrievers.hybrid import hybrid_then_rerank

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.ollama, pytest.mark.chroma]

# Phrases that signal the model is declining. Lenient by design: the
# prompt mandates one exact sentence, but at temperature 0 llama3.2 still
# paraphrases ("I cannot answer ...", "The context does not contain ...")
# often enough that gating on the exact string would flake for wording,
# not for substance.
_REFUSAL_MARKERS = (
    "i don't know",
    "i do not know",
    "don't know based on",
    "do not know based on",
    "cannot answer",
    "can't answer",
    "does not contain",
    "doesn't contain",
    "not contain the answer",
    "no information",
    "not enough information",
    "not provided in the context",
)


def looks_like_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(marker in low for marker in _REFUSAL_MARKERS)


# Rows where llama3.2 reproducibly FAILS to refuse (documented finding, not
# a regression). ooc-002 (plugin authoring): measured 5/5 runs at temperature
# 0, the model fabricates a full plugin-authoring answer, complete with
# invented API and a citation to `how-to__plugin::registering-a-plugin::0`, a
# chunk id that does not exist. This is the adjacent-absent failure mode the
# refusal category exists to surface: a real pytest topic, absent from our 12
# pages, where the model's strong prior plus a near-miss retrieved chunk
# overpower the "say I don't know" instruction. strict=True so this XPASS-fails
# the moment the model starts refusing (a real change worth re-examining the
# finding for), exactly the gen-010 treatment in test_generation_quality.
_XFAIL_REFUSAL = {
    "ooc-002": "S5 finding: model fabricates a plugin answer + a non-existent "
    "source instead of refusing (5/5 runs, temperature 0)",
}


@pytest.fixture(scope="module")
def provider() -> OllamaProvider:
    # Generous timeout: a non-refusing answer can be long (the ooc-002
    # fabrication runs ~2k chars) and occasionally exceeds the 60s default.
    return OllamaProvider(temperature=0.0, timeout=240.0)


def _ids(entries: list[dict]) -> list[str]:
    return [e["id"] for e in entries]


@pytest.fixture
def ooc_entry(request, out_of_corpus_entries):
    return next(e for e in out_of_corpus_entries if e["id"] == request.param)


def _param_for(oid: str):
    marks = []
    if oid in _XFAIL_REFUSAL:
        marks.append(pytest.mark.xfail(reason=_XFAIL_REFUSAL[oid], strict=True))
    return pytest.param(oid, marks=marks)


def pytest_generate_tests(metafunc):
    if "ooc_entry" in metafunc.fixturenames:
        from tests.refusal.conftest import OUT_OF_CORPUS_SET
        from llm_rag.dataset import load_yaml

        entries = load_yaml(OUT_OF_CORPUS_SET)["entries"]
        metafunc.parametrize("ooc_entry", [_param_for(o) for o in _ids(entries)], indirect=True)


async def test_refuses_out_of_corpus(
    ooc_entry, provider, chroma_collection, bm25_index, reranker
):
    query = ooc_entry["query"]

    chunks = hybrid_then_rerank(
        query=query,
        k=5,
        collection=chroma_collection,
        bm25_index=bm25_index,
        reranker=reranker,
    )
    result = await generate_answer(query, chunks, provider)

    logger.info("entry=%s category=%s", ooc_entry["id"], ooc_entry["category"])
    logger.info("query: %s", query)
    logger.info("retrieved (near-miss) chunk ids: %s", [c.id for c in chunks])
    logger.info("prompt:\n%s", build_prompt(query, chunks))
    logger.info("answer:\n%s", result.answer)
    logger.info("cited chunk ids (filtered to index): %s", result.cited_chunk_ids)

    # RAW bracket ids, not the filtered cited_chunk_ids. An out-of-corpus
    # answer often cites a FABRICATED chunk id (e.g. ooc-002 cited
    # `how-to__plugin::registering-a-plugin::0`, which is not in the
    # index) - parse_citations drops that, so the filtered list looks
    # empty and hides the strongest evidence the model did not refuse.
    # parse_all_citations keeps it. See notes.md S5 finding.
    raw_cited = parse_all_citations(result.answer)
    logger.info("raw bracketed ids: %s", raw_cited)

    refused = looks_like_refusal(result.answer)
    # Any bracketed id, real or fabricated, means the model tried to
    # answer from a "source", which for an out-of-corpus question is a
    # hallucinated answer regardless of any hedging text around it.
    assert not raw_cited, (
        f"{ooc_entry['id']}: refusal must cite nothing, got bracketed ids {raw_cited}"
    )
    assert refused, (
        f"{ooc_entry['id']} ({ooc_entry['category']}): expected a refusal, "
        f"model answered instead: {result.answer!r}"
    )
