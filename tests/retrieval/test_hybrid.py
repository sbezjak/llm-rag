"""Hybrid (RRF) retrieval over the full retrieval set.

Two assertions per entry:
  1. The acceptable chunk id appears in top-k.
  2. Hybrid recall is at least as good as the best of vector and
     BM25 on this entry. If hybrid loses to one of its inputs, RRF
     is hurting on that query and that's worth knowing.

Known hybrid misses are encoded as `xfail(strict=True)` in
`KNOWN_HYBRID_XFAILS` below. The "not worse" assertion has no xfails:
if vector + BM25 both miss a chunk, hybrid missing it too is the
expected behavior, not a regression.
"""

from __future__ import annotations

import logging

import pytest
import yaml

from llm_rag.dataset import load_yaml
from llm_rag.retrievers.bm25 import bm25_retrieve
from llm_rag.retrievers.hybrid import hybrid_retrieve
from llm_rag.retrievers.vector import vector_retrieve

from .conftest import RETRIEVAL_SET, expected_ids

logger = logging.getLogger(__name__)

K_VALUES = (5, 10)

KNOWN_HYBRID_XFAILS: dict[tuple[str, int], str] = {
    ("ret-007", 5): (
        "Bare-identifier query `pytest.warns`: both vector and BM25 miss the "
        "right chunk (see test_vector / test_bm25), so RRF over their outputs "
        "cannot recover it. The reranker also misses since the chunk is not in "
        "the hybrid candidate set. Real finding: no retrieval combination here "
        "rescues a bare-identifier query when both signals are weak."
    ),
    ("ret-007", 10): (
        "Same root cause as ret-007 @5: hybrid candidate set does not include "
        "the target chunk because neither input retriever surfaces it."
    ),
}


def _any_hit(ids: list[str], acceptable: list[str]) -> bool:
    return any(a in ids for a in acceptable)


def _params():
    out = []
    for entry in load_yaml(RETRIEVAL_SET)["entries"]:
        for k in K_VALUES:
            reason = KNOWN_HYBRID_XFAILS.get((entry["id"], k))
            marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
            out.append(pytest.param(entry, k, marks=marks, id=f"{entry['id']}-{k}"))
    return out


def _not_worse_params():
    """The 'not worse than inputs' assertion never xfails: if both inputs
    miss, hybrid missing too is correct behavior, not a regression."""
    return [
        pytest.param(entry, k, id=f"{entry['id']}-{k}")
        for entry in load_yaml(RETRIEVAL_SET)["entries"]
        for k in K_VALUES
    ]


@pytest.mark.chroma
@pytest.mark.parametrize("entry,k", _params())
def test_hybrid_recall(entry, k, chroma_collection, bm25_index):
    expected = expected_ids(entry)
    chunks = hybrid_retrieve(
        entry["query"], k=k, collection=chroma_collection, bm25_index=bm25_index
    )
    got_ids = [c.id for c in chunks]

    logger.info("entry=%s category=%s k=%d", entry["id"], entry["category"], k)
    logger.info("query: %s", entry["query"])
    logger.info("acceptable ids: %s", expected)
    logger.info("top-%d ids:\n%s", k, yaml.safe_dump(got_ids, sort_keys=False))

    assert _any_hit(got_ids, expected), (
        f"{entry['id']} ({entry['category']}): none of {expected} in hybrid top-{k}, got {got_ids}"
    )


@pytest.mark.chroma
@pytest.mark.parametrize("entry,k", _not_worse_params())
def test_hybrid_not_worse_than_inputs(entry, k, chroma_collection, bm25_index):
    """Hybrid should not lose to either of its inputs on a given entry.

    If vector or BM25 found an acceptable chunk in top-k but hybrid
    didn't, RRF is reordering the win out. That's the finding we
    want surfaced rather than averaged into a global rate.
    """
    expected = expected_ids(entry)

    vec = [c.id for c in vector_retrieve(entry["query"], k=k, collection=chroma_collection)]
    bm = [c.id for c in bm25_retrieve(entry["query"], k=k, index=bm25_index)]
    hyb = [
        c.id
        for c in hybrid_retrieve(
            entry["query"], k=k, collection=chroma_collection, bm25_index=bm25_index
        )
    ]

    vec_hit = _any_hit(vec, expected)
    bm_hit = _any_hit(bm, expected)
    hyb_hit = _any_hit(hyb, expected)

    logger.info(
        "entry=%s k=%d vec=%s bm25=%s hybrid=%s", entry["id"], k, vec_hit, bm_hit, hyb_hit
    )

    if vec_hit or bm_hit:
        assert hyb_hit, (
            f"{entry['id']} k={k}: vector_hit={vec_hit} bm25_hit={bm_hit} but hybrid missed; "
            f"RRF lost a win that one of the inputs already had"
        )
