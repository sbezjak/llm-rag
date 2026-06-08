"""Shared fixtures for retrieval tests.

The Chroma collection and its sentence-transformer embedding function
are session-scoped: loading the embedder takes ~10s on a cold cache,
and we don't want to pay that per test.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
import pytest
from chromadb.utils import embedding_functions

from llm_rag.dataset import load_yaml
from llm_rag.retrievers.bm25 import BM25Index, build_bm25

ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = ROOT / ".chroma"
COLLECTION = "pytest_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
RETRIEVAL_SET = ROOT / "data" / "retrieval_set.yaml"


@pytest.fixture(scope="session")
def chroma_collection():
    if not CHROMA_DIR.exists():
        pytest.skip("no chroma index, run `uv run python -m llm_rag.scripts.build_index`")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_collection(COLLECTION, embedding_function=ef)


@pytest.fixture(scope="session")
def retrieval_entries() -> list[dict]:
    return load_yaml(RETRIEVAL_SET)["entries"]


@pytest.fixture(scope="session")
def bm25_index(chroma_collection) -> BM25Index:
    """Build a BM25 index from the same chunks that Chroma holds.

    Same source of truth as the vector retriever: identical ids,
    identical text. The only difference between the two retrievers is
    the scoring function.
    """
    raw = chroma_collection.get(include=["documents", "metadatas"])
    chunks = [
        {
            "id": cid,
            "text": doc,
            "source": (meta or {}).get("source", ""),
            "section": (meta or {}).get("section", ""),
        }
        for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])
    ]
    return build_bm25(chunks)


def expected_ids(entry: dict) -> list[str]:
    if "acceptable_chunk_ids" in entry:
        return list(entry["acceptable_chunk_ids"])
    return [entry["expected_chunk_id"]]
