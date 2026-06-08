"""Hybrid retriever: reciprocal rank fusion of vector + BM25.

RRF (Cormack et al. 2009) combines two ranked lists by summing
1 / (k_rrf + rank) for each chunk that appears in either list. Chunks
that both retrievers rank highly win; chunks that only one retriever
likes can still surface if the other doesn't actively rank them low.

Why RRF instead of weighted score blending: vector cosine and BM25
scores live on different scales (0..1 vs 0..40+). Mixing raw scores
would require calibration. Ranks are unit-free and comparable across
retrievers without tuning.

k_rrf = 60 is the constant from the original paper. Robust enough
that retuning it rarely helps.
"""

from __future__ import annotations

import logging

from .bm25 import BM25Index, bm25_retrieve
from .reranker import Reranker
from .vector import RetrievedChunk, vector_retrieve

logger = logging.getLogger(__name__)

K_RRF = 60


def hybrid_retrieve(
    query: str,
    k: int,
    collection,
    bm25_index: BM25Index,
    candidate_k: int = 50,
) -> list[RetrievedChunk]:
    """Retrieve top-k by RRF over vector + BM25.

    Each underlying retriever is asked for `candidate_k` results; RRF
    fuses the two lists, then we return the top-k of the fused
    ranking. candidate_k should be larger than k so that chunks the
    two retrievers disagree on still have a chance to be combined.
    """
    vec_hits = vector_retrieve(query, k=candidate_k, collection=collection)
    bm_hits = bm25_retrieve(query, k=candidate_k, index=bm25_index)

    chunk_by_id: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}
    for rank, c in enumerate(vec_hits, start=1):
        chunk_by_id.setdefault(c.id, c)
        scores[c.id] = scores.get(c.id, 0.0) + 1.0 / (K_RRF + rank)
    for rank, c in enumerate(bm_hits, start=1):
        chunk_by_id.setdefault(c.id, c)
        scores[c.id] = scores.get(c.id, 0.0) + 1.0 / (K_RRF + rank)

    fused_ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:k]
    fused = [
        RetrievedChunk(
            id=cid,
            text=chunk_by_id[cid].text,
            score=scores[cid],
            source=chunk_by_id[cid].source,
            section=chunk_by_id[cid].section,
        )
        for cid in fused_ids
    ]

    logger.info("hybrid_retrieve query=%r k=%d candidate_k=%d", query, k, candidate_k)
    for rank, c in enumerate(fused, start=1):
        logger.info("  rank=%d rrf_score=%.4f id=%s", rank, c.score, c.id)

    return fused


def hybrid_then_rerank(
    query: str,
    k: int,
    collection,
    bm25_index: BM25Index,
    reranker: Reranker,
    candidate_k: int = 25,
) -> list[RetrievedChunk]:
    """Hybrid fuse to `candidate_k`, then cross-encoder rerank to `k`.

    candidate_k defaults to 25: cross-encoder scoring is O(candidates)
    in model calls, and recall gains flatten past ~25 for a small
    corpus. Bump it if the underlying hybrid recall@candidate_k is
    leaving expected chunks out before the reranker sees them.
    """
    candidates = hybrid_retrieve(
        query=query,
        k=candidate_k,
        collection=collection,
        bm25_index=bm25_index,
        candidate_k=candidate_k,
    )
    return reranker.rerank(query=query, candidates=candidates, k=k)
