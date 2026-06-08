"""Vector retriever: embed the query, ask Chroma for nearest chunks.

Pure function over (query, k, collection). No LLM call, no global
state. The caller owns the Chroma collection so tests can swap it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    source: str
    section: str


def vector_retrieve(query: str, k: int, collection) -> list[RetrievedChunk]:
    res = collection.query(query_texts=[query], n_results=k)
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    chunks = [
        RetrievedChunk(
            id=cid,
            text=doc,
            score=1.0 - dist,
            source=(meta or {}).get("source", ""),
            section=(meta or {}).get("section", ""),
        )
        for cid, doc, meta, dist in zip(ids, docs, metas, dists)
    ]

    logger.info("vector_retrieve query=%r k=%d", query, k)
    for rank, c in enumerate(chunks, start=1):
        logger.info("  rank=%d score=%.4f id=%s", rank, c.score, c.id)

    return chunks
