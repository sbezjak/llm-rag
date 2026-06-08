"""Capture the day-1 observability baseline to reports/baseline_day1.json.

This is the additive S6 baseline, NOT a replacement for the S3 retrieval
baseline (reports/retrieval_baseline.json, recall@k). That one answers
"does retrieval fetch the right chunk?"; this one answers "what does the
generation path cost and how good is it RIGHT NOW?", the fixed numbers
future sessions diff against. Per PLAN.md, without this baseline the
observability work is decoration.

What it measures, over the generation set (known-good context, so
quality is not contaminated by retrieval, the same separation the
generation tests keep):

- generation latency p50 / p95 (the provider call)
- tokens per query (prompt + completion, from Ollama's eval counts)
- scorer means (semantic, rouge, judge) of the model answer vs the
  hand-written expected_answer

Retrieval latency is measured separately, by timing hybrid_then_rerank
over each query. It is recorded for completeness but deliberately does
NOT feed the generation answer: the generator still gets the blessed
known-good context, so a retrieval miss cannot move the quality numbers.

Requires a live Ollama and a built Chroma index. Run:

    uv run python -m llm_rag.scripts.run_day1_baseline
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date
from pathlib import Path
from statistics import mean

import chromadb
from chromadb.utils import embedding_functions

from llm_rag.dataset import load_yaml
from llm_rag.generators.generate import build_prompt
from llm_rag.observability import get_tracer, trace_generation, trace_retrieval
from llm_rag.providers.ollama import OllamaProvider
from llm_rag.retrievers.bm25 import build_bm25
from llm_rag.retrievers.hybrid import hybrid_then_rerank
from llm_rag.retrievers.reranker import default_reranker
from llm_rag.retrievers.vector import RetrievedChunk
from llm_rag.scorers.judge import LLMJudgeScorer
from llm_rag.scorers.rouge import RougeScorer
from llm_rag.scorers.semantic import SemanticScorer

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = ROOT / ".chroma"
COLLECTION = "pytest_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
GENERATION_SET = ROOT / "data" / "generation_set.yaml"
REPORT_PATH = ROOT / "reports" / "baseline_day1.json"
TRACE_PATH = ROOT / "reports" / "baseline_day1_trace.json"
MODEL = "llama3.2"
RETRIEVE_K = 5
RERANK_CANDIDATE_K = 25


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Small n (10 rows), so no interpolation
    games, the nearest-rank definition is the honest one to report and
    matches what a reader would compute by hand."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(pct / 100 * len(ordered))))
    return round(ordered[rank - 1], 4)


def _chunks_for(entry: dict) -> list[RetrievedChunk]:
    """Known-good context as RetrievedChunks (mirrors the generation
    tests' fixture: only id + text matter to the generator)."""
    cid = entry["context_chunk_ids"][0]
    return [
        RetrievedChunk(
            id=cid,
            text=entry["known_good_context"],
            score=1.0,
            source=cid.split("::", 1)[0],
            section=cid.split("::")[1] if "::" in cid else "",
        )
    ]


def _open_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_collection(COLLECTION, embedding_function=ef)


def _load_chunks(collection) -> list[dict]:
    raw = collection.get(include=["documents", "metadatas"])
    return [
        {
            "id": cid,
            "text": doc,
            "source": (meta or {}).get("source", ""),
            "section": (meta or {}).get("section", ""),
        }
        for cid, doc, meta in zip(raw["ids"], raw["documents"], raw["metadatas"])
    ]


async def capture_report(trace_path: Path | None = None) -> dict:
    """Run the generation set, return the baseline report dict.

    Shared by the day-1 baseline and the day-7 drift check so BOTH runs
    measure with identical code, a diff between two runs is only honest
    if the measurement path is the same. The day-1 script writes this to
    baseline_day1.json; the drift script captures a fresh one and diffs.
    Optionally flushes the raw trace spans to trace_path.
    """
    entries = [
        e for e in load_yaml(GENERATION_SET)["entries"] if e.get("expected_answer") != "TODO"
    ]
    collection = _open_collection()
    bm25_index = build_bm25(_load_chunks(collection))
    reranker = default_reranker()

    tracer = get_tracer()

    # Pass 1: clean retrieval-latency pass, BEFORE any scorer models load
    # or any generation runs. Placement matters: the first cut interleaved
    # retrieval timing with the generation + scoring loop and measured
    # 35-189s per query; in isolation retrieval is ~2s (the cross-encoder
    # reranker is ~90% of it). The inflation was resource contention from
    # the semantic scorer's transformer and the judge's second Ollama call
    # churning alongside, not retrieval cost. The warm-up call pays the
    # one-time embedder / reranker load so it does not land on query 1.
    hybrid_then_rerank(
        query=entries[0]["query"],
        k=RETRIEVE_K,
        collection=collection,
        bm25_index=bm25_index,
        reranker=reranker,
        candidate_k=RERANK_CANDIDATE_K,
    )
    retrieval_latency: dict[str, float] = {}
    for entry in entries:
        query = entry["query"]
        trace_retrieval(
            tracer,
            "retrieval",
            query,
            lambda q=query: hybrid_then_rerank(
                query=q,
                k=RETRIEVE_K,
                collection=collection,
                bm25_index=bm25_index,
                reranker=reranker,
                candidate_k=RERANK_CANDIDATE_K,
            ),
        )
        retrieval_latency[entry["id"]] = tracer.spans[-1].duration_s

    # Pass 2: generation on the BLESSED context (quality stays
    # uncontaminated by retrieval) plus the three scorers.
    provider = OllamaProvider(model=MODEL, temperature=0.0)
    semantic = SemanticScorer()
    rouge = RougeScorer()
    judge = LLMJudgeScorer()

    rows: list[dict] = []
    for entry in entries:
        query = entry["query"]
        expected = entry["expected_answer"]
        ctx = _chunks_for(entry)
        prompt = build_prompt(query, ctx)
        result = await trace_generation(tracer, query, prompt, lambda: provider.complete(prompt))
        generation_latency = tracer.spans[-1].duration_s
        answer = result.text

        sem = await semantic.score(query, answer, expected)
        rou = await rouge.score(query, answer, expected)
        jud = await judge.score(query, answer, expected)

        prompt_tokens = result.prompt_tokens
        completion_tokens = result.completion_tokens
        rows.append(
            {
                "id": entry["id"],
                "query": query,
                "retrieval_latency_s": round(retrieval_latency[entry["id"]], 4),
                "generation_latency_s": round(generation_latency, 4),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": (
                    (prompt_tokens or 0) + (completion_tokens or 0)
                    if prompt_tokens is not None or completion_tokens is not None
                    else None
                ),
                "scores": {
                    "semantic": round(sem.score, 4),
                    "rouge": round(rou.score, 4),
                    "judge": round(jud.score, 4),
                },
            }
        )

    gen_latencies = [r["generation_latency_s"] for r in rows]
    ret_latencies = [r["retrieval_latency_s"] for r in rows]
    total_tokens = [r["total_tokens"] for r in rows if r["total_tokens"] is not None]

    report = {
        "captured": date.today().isoformat(),
        "model": MODEL,
        "n_queries": len(rows),
        "query_set": "generation_set",
        "tracing_backend": tracer.backend,
        "generation_latency_s": {
            "p50": _percentile(gen_latencies, 50),
            "p95": _percentile(gen_latencies, 95),
            "mean": round(mean(gen_latencies), 4) if gen_latencies else 0.0,
        },
        "retrieval_latency_s": {
            "p50": _percentile(ret_latencies, 50),
            "p95": _percentile(ret_latencies, 95),
            "mean": round(mean(ret_latencies), 4) if ret_latencies else 0.0,
        },
        "tokens_per_query": {
            "mean_total": round(mean(total_tokens), 2) if total_tokens else None,
            "mean_prompt": round(
                mean([r["prompt_tokens"] for r in rows if r["prompt_tokens"] is not None]), 2
            )
            if any(r["prompt_tokens"] is not None for r in rows)
            else None,
            "mean_completion": round(
                mean([r["completion_tokens"] for r in rows if r["completion_tokens"] is not None]),
                2,
            )
            if any(r["completion_tokens"] is not None for r in rows)
            else None,
        },
        "scorer_means": {
            "semantic": round(mean([r["scores"]["semantic"] for r in rows]), 4) if rows else 0.0,
            "rouge": round(mean([r["scores"]["rouge"] for r in rows]), 4) if rows else 0.0,
            "judge": round(mean([r["scores"]["judge"] for r in rows]), 4) if rows else 0.0,
        },
        "per_query": rows,
    }

    if trace_path is not None:
        tracer.flush(trace_path)
    return report


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    report = await capture_report(trace_path=TRACE_PATH)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("wrote %s", REPORT_PATH)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
