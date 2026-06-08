import logging
from pathlib import Path

import chromadb
import pytest

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT / ".chroma"
COLLECTION = "pytest_docs"


@pytest.mark.chroma
def test_index_has_chunks():
    if not CHROMA_DIR.exists():
        pytest.skip("no chroma index, run `uv run python -m llm_rag.scripts.build_index`")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    coll = client.get_collection(COLLECTION)
    count = coll.count()
    sample_ids = coll.get(limit=5)["ids"]
    logger.info("collection %r at %s: %d chunks", COLLECTION, CHROMA_DIR, count)
    logger.info("sample chunk ids: %s", sample_ids)
    assert count >= 30, f"expected >=30 chunks, got {count}"
