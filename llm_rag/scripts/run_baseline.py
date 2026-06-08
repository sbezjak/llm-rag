"""Compute the retrieval baseline and write it to reports/.

Per category and overall: recall@1, recall@5, recall@10 for vector,
BM25, hybrid, and hybrid+reranker. This is the snapshot that future
sessions compare against. The point is to have a fixed number to
diff against, not just test pass/fail.

recall@1 is the top-1 precision metric: an acceptable chunk is the
very first result. It moves independently from recall@5 (the right
chunk is anywhere in top-5) and is the signal that the reranker is
actually doing useful reordering.

Usage:
    uv run python -m llm_rag.scripts.run_baseline
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from llm_rag.dataset import load_yaml
from llm_rag.retrievers.bm25 import build_bm25, bm25_retrieve
from llm_rag.retrievers.hybrid import hybrid_retrieve, hybrid_then_rerank
from llm_rag.retrievers.reranker import default_reranker
from llm_rag.retrievers.vector import vector_retrieve

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = ROOT / ".chroma"
COLLECTION = "pytest_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
RETRIEVAL_SET = ROOT / "data" / "retrieval_set.yaml"
REPORT_PATH = ROOT / "reports" / "retrieval_baseline.json"

K_VALUES = [1, 5, 10]
RERANK_CANDIDATE_K = 25


def _expected(entry: dict) -> list[str]:
    if "acceptable_chunk_ids" in entry:
        return list(entry["acceptable_chunk_ids"])
    return [entry["expected_chunk_id"]]


def _hit(ids: list[str], expected: list[str]) -> bool:
    return any(e in ids for e in expected)


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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    entries = load_yaml(RETRIEVAL_SET)["entries"]
    collection = _open_collection()
    bm25_index = build_bm25(_load_chunks(collection))
    reranker = default_reranker()

    retrievers = ("vector", "bm25", "hybrid", "rerank")
    # results[retriever][k][entry_id] = bool
    results: dict[str, dict[int, dict[str, bool]]] = {
        r: {k: {} for k in K_VALUES} for r in retrievers
    }
    categories: dict[str, str] = {e["id"]: e["category"] for e in entries}

    for entry in entries:
        expected = _expected(entry)
        for k in K_VALUES:
            vec = [c.id for c in vector_retrieve(entry["query"], k=k, collection=collection)]
            bm = [c.id for c in bm25_retrieve(entry["query"], k=k, index=bm25_index)]
            hyb = [
                c.id
                for c in hybrid_retrieve(
                    entry["query"], k=k, collection=collection, bm25_index=bm25_index
                )
            ]
            rer = [
                c.id
                for c in hybrid_then_rerank(
                    entry["query"],
                    k=k,
                    collection=collection,
                    bm25_index=bm25_index,
                    reranker=reranker,
                    candidate_k=RERANK_CANDIDATE_K,
                )
            ]
            results["vector"][k][entry["id"]] = _hit(vec, expected)
            results["bm25"][k][entry["id"]] = _hit(bm, expected)
            results["hybrid"][k][entry["id"]] = _hit(hyb, expected)
            results["rerank"][k][entry["id"]] = _hit(rer, expected)

    # Aggregate per category and overall.
    by_category: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    overall: dict[str, dict[str, float]] = defaultdict(dict)
    per_category_ids: dict[str, list[str]] = defaultdict(list)
    for eid, cat in categories.items():
        per_category_ids[cat].append(eid)

    for retriever, by_k in results.items():
        for k, hits in by_k.items():
            for cat, ids in per_category_ids.items():
                hit_count = sum(hits[eid] for eid in ids)
                by_category[cat][retriever][f"recall@{k}"] = round(hit_count / len(ids), 4)
            overall[retriever][f"recall@{k}"] = round(sum(hits.values()) / len(hits), 4)

    misses: dict[str, dict[int, list[str]]] = {
        r: {k: [eid for eid, hit in by_k[k].items() if not hit] for k in K_VALUES}
        for r, by_k in results.items()
    }

    report = {
        "k_values": K_VALUES,
        "rerank_candidate_k": RERANK_CANDIDATE_K,
        "n_entries": len(entries),
        "by_category": {
            cat: dict(by_category[cat]) for cat in sorted(per_category_ids)
        },
        "overall": dict(overall),
        "misses_by_retriever": misses,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    logger.info("wrote %s", REPORT_PATH)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
