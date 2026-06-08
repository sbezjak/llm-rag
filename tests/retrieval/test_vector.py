"""Vector retrieval over the full retrieval set.

For each (entry, k), embed the query, fetch top-k from the Chroma
index, and assert that at least one acceptable chunk id appears in
top-k. Failure means vector retrieval missed the answer chunk for
that query at that k. The category (paraphrase / identifier / mixed
/ tough) tells you which failure modes vector struggles with.

Known vector misses are encoded as `xfail(strict=True)` in
`KNOWN_VECTOR_XFAILS` below, so a silent recovery (the model or the
index changed and the miss disappeared) flips red and forces a
re-look instead of being averaged into the global recall number.
"""

from __future__ import annotations

import logging

import pytest
import yaml

from llm_rag.dataset import load_yaml
from llm_rag.retrievers.vector import vector_retrieve

from .conftest import RETRIEVAL_SET, expected_ids

logger = logging.getLogger(__name__)

K_VALUES = (5, 10)

# Per-(entry_id, k) reasons for documented vector misses. Keep the
# reason concrete: name what the retriever returned instead and why.
KNOWN_VECTOR_XFAILS: dict[tuple[str, int], str] = {
    ("ret-007", 5): (
        "Bare-identifier query `pytest.warns` returns warnings/exceptions/skipping "
        "chunks instead of the dedicated `assertions-about-expected-warnings::0`. "
        "Embedding has too little context (two tokens, no sentence) to lean on; "
        "hybrid + reranker also miss (see test_bm25 / test_hybrid / test_reranker)."
    ),
    ("ret-007", 10): (
        "Same root cause as ret-007 @5: bare-identifier `pytest.warns` is not in "
        "vector top-10 either. The chunk exists and contains the literal text; "
        "the embedding just cannot rank it without sentence context."
    ),
    ("ret-012", 5): (
        "Intent-ambiguous query: vector returns how-to__mark chunks (attach a "
        "marker to a test) instead of how-to__usage::specifying-which-tests-to-run "
        "(select tests by marker). Hybrid+reranker recover it; pure vector misses "
        "at top-5. Documented as expected vector limitation."
    ),
    ("ret-014", 5): (
        "Same root cause as ret-012 but with a literal `pytest -m` query: vector "
        "still picks how-to__mark chunks at top-5. The chunk is in vector top-10, "
        "so it's a ranking issue not a recall issue. Hybrid surfaces it correctly."
    ),
}


def _params():
    """Flatten (entry, k) into a single parametrize so we can attach
    per-(entry, k) xfail marks. Mirrors the pattern in test_reranker.py."""
    out = []
    for entry in load_yaml(RETRIEVAL_SET)["entries"]:
        for k in K_VALUES:
            reason = KNOWN_VECTOR_XFAILS.get((entry["id"], k))
            marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
            out.append(pytest.param(entry, k, marks=marks, id=f"{entry['id']}-{k}"))
    return out


@pytest.mark.chroma
@pytest.mark.parametrize("entry,k", _params())
def test_vector_recall(entry, k, chroma_collection):
    expected = expected_ids(entry)
    chunks = vector_retrieve(entry["query"], k=k, collection=chroma_collection)
    got_ids = [c.id for c in chunks]

    logger.info("entry=%s category=%s k=%d", entry["id"], entry["category"], k)
    logger.info("query: %s", entry["query"])
    logger.info("acceptable ids: %s", expected)
    logger.info("top-%d ids:\n%s", k, yaml.safe_dump(got_ids, sort_keys=False))

    assert any(e in got_ids for e in expected), (
        f"{entry['id']} ({entry['category']}): none of {expected} in top-{k}, got {got_ids}"
    )
