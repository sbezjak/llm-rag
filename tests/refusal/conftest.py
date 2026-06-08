"""Fixtures for refusal tests.

Refusal is a whole-pipeline property: the question goes through real
retrieval (hybrid + rerank) and a live model, because the thing under
test is whether the model declines when the corpus does not contain the
answer. So unlike generation tests (which inject known-good context),
refusal tests do NOT bypass retrieval, they feed the model whatever the
real retriever surfaces for an out-of-corpus query, which is the whole
point: a near-miss chunk is exactly what tempts a hallucinated answer.

The chroma collection, BM25 index, and reranker are session-scoped for
the usual reason: loading the embedder and the cross-encoder is slow and
we don't want to pay it per test. These mirror tests/retrieval/conftest
deliberately rather than sharing a module: the two suites are
independent and a change to one should not silently move the other.
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
