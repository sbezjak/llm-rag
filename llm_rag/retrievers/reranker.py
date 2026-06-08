"""Cross-encoder reranker on top of hybrid candidates.

A cross-encoder (here `BAAI/bge-reranker-base`) scores `(query, chunk)`
as a pair, attending to both jointly. Bi-encoders (the vector
retriever) embed query and chunk independently and then compare, which
is fast but loses interaction signal. Cross-encoders are too slow to
run over the whole corpus, so the pipeline is: hybrid gets you ~25-50
plausible candidates cheaply, the reranker reorders them.

The reranker is wrapped in a class so tests can inject a fake. A
module-level `default_reranker()` returns a cached singleton for
non-test code, since the model is ~270MB and loading it per call is
wasteful.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from functools import lru_cache
from typing import Protocol

from .vector import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-base"


class CrossEncoderLike(Protocol):
    def predict(self, pairs: list[list[str]]) -> list[float]: ...


class Reranker:
    """Rerank candidate chunks by cross-encoder score.

    The score field on the returned chunks is overwritten with the
    rerank score so downstream code sorts by the new signal. Original
    hybrid scores are dropped on purpose: keeping both would invite
    callers to mix scales that don't compare.
    """

    def __init__(self, model: CrossEncoderLike, model_name: str = DEFAULT_MODEL) -> None:
        self._model = model
        self._model_name = model_name

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        k: int,
    ) -> list[RetrievedChunk]:
        if not candidates:
            logger.info("rerank query=%r candidates=0", query)
            return []

        pairs = [[query, c.text] for c in candidates]
        raw_scores = self._model.predict(pairs)
        scores = [float(s) for s in raw_scores]

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        top = scored[:k]

        logger.info(
            "rerank model=%s query=%r candidates=%d k=%d",
            self._model_name,
            query,
            len(candidates),
            k,
        )
        for rank, (chunk, score) in enumerate(scored, start=1):
            logger.info("  rank=%d rerank_score=%.4f id=%s", rank, score, chunk.id)

        return [replace(chunk, score=score) for chunk, score in top]


@lru_cache(maxsize=1)
def default_reranker(model_name: str = DEFAULT_MODEL) -> Reranker:
    """Load the bge cross-encoder once and reuse it.

    Trade-off: a module-level cache means the model lives for the
    process lifetime. Production code would inject the Reranker
    explicitly so its lifecycle is visible; for this project the
    singleton keeps scripts and the (few) real-model tests cheap.
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    return Reranker(model=model, model_name=model_name)
