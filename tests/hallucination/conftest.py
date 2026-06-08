"""Fixtures for hallucinated-source tests.

The live test needs the set of every chunk id that actually exists in
the index, that set is the ground truth a citation is hallucinated
against. Session-scoped: one read of the collection ids serves all
parametrised rows.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
import pytest
from chromadb.utils import embedding_functions

from llm_rag.dataset import load_yaml
from llm_rag.retrievers.bm25 import BM25Index, build_bm25
from llm_rag.retrievers.reranker import default_reranker

ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = ROOT / ".chroma"
COLLECTION = "pytest_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
OUT_OF_CORPUS_SET = ROOT / "data" / "out_of_corpus.yaml"


@pytest.fixture(scope="session")
def chroma_collection():
    if not CHROMA_DIR.exists():
        pytest.skip("no chroma index, run `uv run python -m llm_rag.scripts.build_index`")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_collection(COLLECTION, embedding_function=ef)


@pytest.fixture(scope="session")
def index_ids(chroma_collection) -> set[str]:
    """Every chunk id that exists in the index."""
    return set(chroma_collection.get()["ids"])


@pytest.fixture(scope="session")
def bm25_index(chroma_collection) -> BM25Index:
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


@pytest.fixture(scope="session")
def reranker():
    return default_reranker()


@pytest.fixture(scope="session")
def out_of_corpus_entries() -> list[dict]:
    return load_yaml(OUT_OF_CORPUS_SET)["entries"]
