"""BM25 retrieval over the full retrieval set.

Mirrors test_vector.py shape so the two are directly comparable: same
entries, same k values, same assertion. The interesting reads are the
diffs between vector and BM25, not either rate in isolation.

Known BM25 misses are encoded as `xfail(strict=True)` in
`KNOWN_BM25_XFAILS` below.
"""

from __future__ import annotations

import logging

import pytest
import yaml

from llm_rag.dataset import load_yaml
from llm_rag.retrievers.bm25 import bm25_retrieve

from .conftest import RETRIEVAL_SET, expected_ids

logger = logging.getLogger(__name__)

K_VALUES = (5, 10)

KNOWN_BM25_XFAILS: dict[tuple[str, int], str] = {
    ("ret-001", 5): (
        "Pure paraphrase query (`how do I write my first pytest test`) has no "
        "discriminating identifier for BM25. The chunk title is `create your "
        "first test`, no shared token weighted highly enough to surface it. "
        "Documented vector/BM25 asymmetry: vector crushes this, BM25 cannot."
    ),
    ("ret-001", 10): (
        "Same root cause as ret-001 @5: BM25 has no identifier overlap on this "
        "paraphrase query. Vector finds it at top-1 (see test_vector); BM25 "
        "alone misses even at top-10."
    ),
    ("ret-007", 5): (
        "Bare-identifier query `pytest.warns`: BM25 tokenizes it to "
        "[pytest, warns]. `pytest` appears in nearly every chunk (no signal); "
        "`warns` does discriminate but the chunk's term frequency is not "
        "enough to lift it over chunks that have `warns` plus other matching "
        "tokens. Vector also misses (see test_vector); reranker cannot recover."
    ),
    ("ret-007", 10): (
        "Same root cause as ret-007 @5: BM25 top-10 still does not include "
        "the dedicated `assertions-about-expected-warnings::0` chunk."
    ),
}


def _params():
    out = []
    for entry in load_yaml(RETRIEVAL_SET)["entries"]:
        for k in K_VALUES:
            reason = KNOWN_BM25_XFAILS.get((entry["id"], k))
            marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
            out.append(pytest.param(entry, k, marks=marks, id=f"{entry['id']}-{k}"))
    return out


@pytest.mark.chroma
@pytest.mark.parametrize("entry,k", _params())
def test_bm25_recall(entry, k, bm25_index):
    expected = expected_ids(entry)
    chunks = bm25_retrieve(entry["query"], k=k, index=bm25_index)
    got_ids = [c.id for c in chunks]

    logger.info("entry=%s category=%s k=%d", entry["id"], entry["category"], k)
    logger.info("query: %s", entry["query"])
    logger.info("acceptable ids: %s", expected)
    logger.info("top-%d ids:\n%s", k, yaml.safe_dump(got_ids, sort_keys=False))

    assert any(e in got_ids for e in expected), (
        f"{entry['id']} ({entry['category']}): none of {expected} in top-{k}, got {got_ids}"
    )
