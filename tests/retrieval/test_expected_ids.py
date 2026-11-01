"""A retrieval miss only means something if the expected chunk is in the index.

ret-007 was pinned as a search failure across every retriever while its target
chunk was missing from the index, dropped by the chunker's minimum-size rule.
"""

import pytest

from .conftest import expected_ids

pytestmark = pytest.mark.chroma


def test_every_expected_chunk_is_in_the_index(chroma_collection, retrieval_entries):
    indexed = set(chroma_collection.get(include=[])["ids"])
    missing = {
        e["id"]: [cid for cid in expected_ids(e) if cid not in indexed] for e in retrieval_entries
    }
    assert not {k: v for k, v in missing.items() if v}
