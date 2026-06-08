"""BM25 retriever: classic sparse lexical retrieval over the same
chunks the vector retriever sees.

BM25 ranks documents by how often the query's terms appear in each
document, with length normalization and a saturation curve so a single
exact term match doesn't dominate. It complements vector retrieval:
strong on exact identifier matches (`@pytest.fixture`, `--cache-clear`),
weak on paraphrase.

Tokenization here is deliberately simple: lowercase, split on
non-alphanumeric, drop empties. Production BM25 often adds stopword
removal and stemming; we skip both so behavior stays inspectable.
Trade-off noted in notes.md.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .vector import RetrievedChunk

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


@dataclass
class BM25Index:
    ids: list[str]
    texts: list[str]
    sources: list[str]
    sections: list[str]
    bm25: BM25Okapi


def build_bm25(chunks: list[dict]) -> BM25Index:
    """Build a BM25 index from chunk dicts: {id, text, source, section}."""
    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    sources = [c.get("source", "") for c in chunks]
    sections = [c.get("section", "") for c in chunks]
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    return BM25Index(ids=ids, texts=texts, sources=sources, sections=sections, bm25=bm25)


def bm25_retrieve(query: str, k: int, index: BM25Index) -> list[RetrievedChunk]:
    scores = index.bm25.get_scores(tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    chunks = [
        RetrievedChunk(
            id=index.ids[i],
            text=index.texts[i],
            score=float(scores[i]),
            source=index.sources[i],
            section=index.sections[i],
        )
        for i in top
    ]

    logger.info("bm25_retrieve query=%r k=%d", query, k)
    for rank, c in enumerate(chunks, start=1):
        logger.info("  rank=%d score=%.4f id=%s", rank, c.score, c.id)

    return chunks
