"""Tests for the cross-encoder reranker, in two layers.

1. Mocked unit tests with a fake CrossEncoder. Confirm the Reranker
   sorts by score, trims to k, overwrites the chunk score, and logs.
   These run without loading the real model.
2. Integration tests over the retrieval set with the real
   `bge-reranker-base`. Two assertions per entry:
     - top-5 recall (an acceptable chunk appears in the top-5), same
       contract as the other retriever tests.
     - top-1 precision (an acceptable chunk is the top result), a
       stricter signal that the reranker is actually reordering.
   These move independently: a query can keep recall (hybrid already
   surfaced the right chunk) but gain or lose top-1 (reranker either
   floats it up or buries it).
"""

from __future__ import annotations

import logging

import pytest
import yaml

from llm_rag.dataset import load_yaml
from llm_rag.retrievers.hybrid import hybrid_then_rerank
from llm_rag.retrievers.reranker import Reranker
from llm_rag.retrievers.vector import RetrievedChunk

from .conftest import RETRIEVAL_SET, expected_ids

logger = logging.getLogger(__name__)


def _entries():
    return load_yaml(RETRIEVAL_SET)["entries"]


def _ids():
    return [e["id"] for e in load_yaml(RETRIEVAL_SET)["entries"]]


# Known top-1 limitations of bge-reranker-base on this corpus.
# Each entry is xfailed strict=True: if the reranker ever starts
# getting one of these right, the test fails loudly and we have to
# look at why (model change, candidate set change, dataset change)
# instead of letting the improvement go unnoticed.
# Known top-5 recall misses: the reranker cannot surface a chunk that
# the underlying hybrid step never produced as a candidate. ret-007's
# bare-identifier `pytest.warns` query is missed by both vector and
# BM25 (see test_vector / test_bm25), so the hybrid candidate set
# doesn't contain the target chunk for the reranker to rescue.
TOP5_XFAILS: dict[str, str] = {
    "ret-007": (
        "Bare-identifier query `pytest.warns`: hybrid candidate set does "
        "not include the target chunk (both vector and BM25 miss it; see "
        "test_vector / test_bm25). Reranker can only reorder what hybrid "
        "produced, so this is structurally unrecoverable here."
    ),
}


TOP1_XFAILS: dict[str, str] = {
    "ret-001": (
        "bge picks parametrize::basic-pytest-generate-tests-example::0 over "
        "getting-started::create-your-first-test. Reranker buries plain-English "
        "walkthrough chunks under identifier-heavy ones."
    ),
    "ret-007": (
        "Bare-identifier query `pytest.warns`: target chunk not in hybrid "
        "candidate set (see TOP5_XFAILS). Top-1 cannot be the target if "
        "top-5 cannot."
    ),
    "ret-009": (
        "bge picks how-to-monkeypatch-mock-modules-and-environments::1 (a "
        "teardown-semantics sibling) over the dedicated "
        "monkeypatching-environment-variables::0 or the umbrella ::2. Same "
        "pattern as ret-008: reranker prefers page-overview siblings for "
        "narrow queries instead of the dedicated subsection."
    ),
    "ret-004": (
        "bge picks basic-pytest-generate-tests-example::2 (the pytest_generate_tests "
        "hook) over pytest-mark-parametrize::0/1. Conflates the two parametrize APIs."
    ),
    "ret-005": (
        "bge picks temporary-directory-location-and-retention::1 (about {num} "
        "retention) over the tmp_path fixture intro chunks. Picks a same-page "
        "sibling that does not answer the query."
    ),
    "ret-008": (
        "bge picks page-intro how-to-mark-test-functions-with-attributes::0 over "
        "the specific registering-marks::0. Over-weights umbrella chunks for "
        "narrow queries."
    ),
    "ret-010": (
        "bge picks accessing-captured-output-from-a-test-function::1 (about "
        "capsysbinary) over setting-capturing-methods-or-disabling-capturing::0. "
        "Lexical 'capture' overlap pulls the wrong sibling."
    ),
    "ret-012": (
        "Query was designed as intent-ambiguous, but ret-014 (unambiguous "
        "companion using literal `pytest -m`) also fails top-1 with the same "
        "shape: bge prefers how-to__mark:: chunks (declaring marks) over "
        "how-to__usage::specifying-which-tests-to-run (selecting by marker) "
        "for any marker-related query. The real finding is corpus structure, "
        "not query ambiguity."
    ),
    "ret-014": (
        "Unambiguous companion to ret-012. Same root cause: bge prefers "
        "how-to__mark::registering-marks::0 over the how-to__usage:: "
        "specifying-which-tests-to-run chunks even when the query literally "
        "says `pytest -m`. Top-5 recall is fine; top-1 is buried."
    ),
    "ret-013": (
        "bge picks cache::usage::0 (about --lf / --last-failed) over "
        "clearing-cache-content::0 (which actually documents --cache-clear). "
        "Prefers page-overview chunk for short identifier queries."
    ),
}


def _marks_for(entry: dict, xfails: dict[str, str]) -> list:
    reason = xfails.get(entry["id"])
    if reason:
        return [pytest.mark.xfail(strict=True, reason=reason)]
    return []


def _top5_params():
    return [
        pytest.param(e, marks=_marks_for(e, TOP5_XFAILS), id=e["id"]) for e in _entries()
    ]


def _top1_params():
    return [
        pytest.param(e, marks=_marks_for(e, TOP1_XFAILS), id=e["id"]) for e in _entries()
    ]


def _any_hit(ids: list[str], acceptable: list[str]) -> bool:
    return any(a in ids for a in acceptable)


class _FakeCrossEncoder:
    """Cross-encoder stub. Returns a preset score per chunk id.

    Used so unit tests don't load the real 270MB model. The score for
    a (query, text) pair is looked up by the leading "id:" prefix in
    the text payload that the tests construct.
    """

    def __init__(self, scores_by_id: dict[str, float]) -> None:
        self._scores = scores_by_id

    def predict(self, pairs: list[list[str]]) -> list[float]:
        out = []
        for _query, text in pairs:
            cid = text.split("|", 1)[0]
            out.append(self._scores[cid])
        return out


def _chunk(cid: str) -> RetrievedChunk:
    return RetrievedChunk(id=cid, text=f"{cid}|body", score=0.0, source="src", section="sec")


@pytest.mark.mocked
def test_reranker_sorts_and_trims():
    candidates = [_chunk("a"), _chunk("b"), _chunk("c"), _chunk("d")]
    model = _FakeCrossEncoder({"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.7})
    reranker = Reranker(model=model, model_name="fake")

    out = reranker.rerank(query="q", candidates=candidates, k=2)

    assert [c.id for c in out] == ["b", "d"]
    assert out[0].score == pytest.approx(0.9)
    assert out[1].score == pytest.approx(0.7)


@pytest.mark.mocked
def test_reranker_overwrites_input_score():
    candidate = RetrievedChunk(id="x", text="x|body", score=42.0, source="s", section="t")
    model = _FakeCrossEncoder({"x": 0.33})
    reranker = Reranker(model=model, model_name="fake")

    out = reranker.rerank(query="q", candidates=[candidate], k=1)

    assert out[0].score == pytest.approx(0.33)


@pytest.mark.mocked
def test_reranker_empty_candidates():
    reranker = Reranker(model=_FakeCrossEncoder({}), model_name="fake")
    assert reranker.rerank(query="q", candidates=[], k=5) == []


@pytest.fixture(scope="session")
def real_reranker():
    """Load the bge cross-encoder once per session.

    Skipped on import error so contributors without the model cache
    can still run the mocked tests.
    """
    try:
        from llm_rag.retrievers.reranker import default_reranker
    except ImportError as exc:
        pytest.skip(f"sentence-transformers unavailable: {exc}")
    return default_reranker()


@pytest.mark.chroma
@pytest.mark.parametrize("entry", _top5_params())
def test_reranker_top5_recall(entry, chroma_collection, bm25_index, real_reranker):
    expected = expected_ids(entry)
    chunks = hybrid_then_rerank(
        entry["query"],
        k=5,
        collection=chroma_collection,
        bm25_index=bm25_index,
        reranker=real_reranker,
        candidate_k=25,
    )
    got_ids = [c.id for c in chunks]

    logger.info("entry=%s category=%s", entry["id"], entry["category"])
    logger.info("query: %s", entry["query"])
    logger.info("acceptable ids: %s", expected)
    logger.info("top-5 ids:\n%s", yaml.safe_dump(got_ids, sort_keys=False))

    assert _any_hit(got_ids, expected), (
        f"{entry['id']} ({entry['category']}): none of {expected} in reranked top-5, got {got_ids}"
    )


@pytest.mark.chroma
@pytest.mark.parametrize("entry", _top1_params())
def test_reranker_top1_precision(entry, chroma_collection, bm25_index, real_reranker):
    expected = expected_ids(entry)
    chunks = hybrid_then_rerank(
        entry["query"],
        k=5,
        collection=chroma_collection,
        bm25_index=bm25_index,
        reranker=real_reranker,
        candidate_k=25,
    )
    top1 = [chunks[0].id] if chunks else []

    logger.info("entry=%s category=%s", entry["id"], entry["category"])
    logger.info("query: %s", entry["query"])
    logger.info("acceptable ids: %s", expected)
    logger.info("top-1: %s", top1)

    assert _any_hit(top1, expected), (
        f"{entry['id']} ({entry['category']}): top-1 was {top1}, none of {expected}"
    )
