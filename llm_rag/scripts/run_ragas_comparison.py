"""S8 stretch: compare Ragas's RAG-aware metrics against this project's
own three scorers, on the same answers.

Why this exists. Ragas (https://docs.ragas.io) is the framework people
reach for to "evaluate a RAG pipeline." This project rolled its own
three scorers (semantic / rouge / judge, ported from the eval harness).
The honest question for the writeup is: what does a purpose-built RAG
eval library measure that our three scorers structurally cannot, and
where does it just overlap them? You only learn that by running both
graders on the *same* generated answers and laying the numbers side by
side. That is the single finding this script is here to produce.

How the comparison stays honest (same answer, same embedding model):

- One answer per query, generated once with the project's own generator
  + OllamaProvider, then graded by BOTH systems. Neither grader sees a
  different answer.
- The model UNDER TEST is always llama3.2: it writes every answer, and
  our own three scorers (including the llama3.2-based LLMJudgeScorer)
  grade it. Only Ragas's internal grader is a more capable model
  (RAGAS_JUDGE_MODEL), because llama3.2 cannot satisfy Ragas's strict
  JSON-schema output contract and NaN's every LLM-graded metric (see the
  RAGAS_JUDGE_MODEL comment). That is a tooling requirement, not a model
  benchmark: nothing about the system being evaluated changes.
- Same embedding model for the similarity metrics: Ragas is handed the
  exact `all-MiniLM-L6-v2` SentenceTransformer that SemanticScorer uses,
  wrapped to the langchain Embeddings interface. So Ragas
  `answer_similarity` and our `semantic` score the same vectors.

Like the day-1 baseline, generation runs on the BLESSED known-good
context from generation_set.yaml, not on live retrieval. That keeps a
retrieval miss from contaminating the quality numbers, the same
separation the generation tests keep. A consequence worth noting in the
writeup: with known-good context, Ragas's context-quality metrics
(context_precision) measure an idealized retriever, not ours.

How the metrics line up with our scorers:

    Ragas metric        | our scorer        | what it measures
    --------------------|-------------------|----------------------------
    answer_similarity   | semantic          | answer vs reference, cosine
    answer_correctness  | judge             | answer vs reference, LLM-graded
    (no equivalent)     | rouge             | lexical n-gram overlap
    faithfulness        | (no equivalent)   | answer grounded in context?
    context_precision   | (no equivalent)   | is the context relevant?

The bottom two rows are the point: faithfulness and context_precision
are RAG-aware (they read the context), and our answer-only scorers
cannot express them. The top two rows are the overlap, and where Ragas's
LLM-graded number inherits the known small-judge weakness (PLAN.md
pre-staked decision 5).

Requires the `ragas` extra, a live Ollama, and a built Chroma index:

    uv sync --extra ragas
    uv run python -m llm_rag.scripts.run_ragas_comparison
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date
from pathlib import Path
from statistics import mean

from llm_rag.dataset import load_yaml
from llm_rag.generators.generate import build_prompt
from llm_rag.providers.ollama import OllamaProvider
from llm_rag.retrievers.vector import RetrievedChunk
from llm_rag.scorers.judge import LLMJudgeScorer
from llm_rag.scorers.rouge import RougeScorer
from llm_rag.scorers.semantic import SemanticScorer

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
GENERATION_SET = ROOT / "data" / "generation_set.yaml"
REPORT_PATH = ROOT / "reports" / "ragas_comparison.json"
MODEL = "llama3.2"
# Ragas's LLM-graded metrics (faithfulness, context_precision, the
# factual half of answer_correctness) prompt for a strict JSON schema
# and then parse it. llama3.2 (3B) cannot honor that contract: even with
# Ollama's format="json" forcing valid JSON syntax, it returns JSON that
# does not match the schema (often echoing the prompt), so every
# LLM-graded metric lands as NaN. A more capable judge is a tooling
# requirement, not a model benchmark: the model UNDER TEST stays
# llama3.2 (it still writes every answer and our own scorers still grade
# with it). Only Ragas's grader is upgraded, the same way production RAG
# eval uses a strong judge model. Same family, one size up, so the
# finding isolates judge capability as the variable.
RAGAS_JUDGE_MODEL = "llama3.1:8b"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Which Ragas metric maps to which of our scorers, for the side-by-side
# table. None means "Ragas-only, our scorers have no equivalent" (the
# RAG-aware ones, which are the whole reason to look at Ragas).
RAGAS_TO_OURS = {
    "answer_similarity": "semantic",
    "answer_correctness": "judge",
    "faithfulness": None,
    "context_precision": None,
}


def _chunks_for(entry: dict) -> list[RetrievedChunk]:
    """Known-good context as a RetrievedChunk (mirrors the generation
    tests + day-1 baseline: only id + text reach the generator)."""
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


def _build_ragas_embeddings():
    """Wrap the same all-MiniLM-L6-v2 model SemanticScorer uses in the
    langchain Embeddings interface, so Ragas's similarity metrics score
    effectively the same vectors our `semantic` scorer does.

    Pinned to CPU on purpose. On a Mac, sentence-transformers defaults to
    the Metal GPU (MPS), and so does the 8B Ragas judge running through
    Ollama. Sharing the GPU's unified memory between both made the
    embedding batch fail with kIOGPUCommandBufferCallbackErrorOutOfMemory
    mid-run, which would NaN answer_similarity (the cleanest comparison
    column) for tooling reasons, not real ones. MiniLM on CPU is fast and
    numerically equivalent (cosine identical to ~1e-6), so the comparison
    stays honest while the GPU is left entirely to the judge. This is a
    fresh CPU instance, not SemanticScorer's cached (MPS) model."""
    from langchain_core.embeddings import Embeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, device="cpu")

    class MiniLMEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return model.encode(list(texts)).tolist()

        def embed_query(self, text: str) -> list[float]:
            return model.encode([text])[0].tolist()

    return LangchainEmbeddingsWrapper(MiniLMEmbeddings())


def _build_ragas_llm():
    """Ragas's grader = RAGAS_JUDGE_MODEL via Ollama, at temperature 0
    for determinism.

    `format="json"` constrains the Ollama decoder to emit syntactically
    valid JSON. It is necessary but not sufficient: with llama3.2 the
    model still produced schema-wrong JSON and every LLM-graded metric
    NaN'd, which is why the judge is RAGAS_JUDGE_MODEL (a more capable
    model) and not MODEL. See the RAGAS_JUDGE_MODEL comment for why this
    is a tooling requirement, not a model benchmark."""
    from langchain_ollama import ChatOllama
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(ChatOllama(model=RAGAS_JUDGE_MODEL, temperature=0.0, format="json"))


async def _generate_and_score_ours(entries: list[dict]) -> list[dict]:
    """Generate one answer per query and grade it with OUR three scorers.

    Returns rows carrying the generated answer so the SAME text is handed
    to Ragas afterwards."""
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
        answer = (await provider.complete(prompt)).text

        sem = await semantic.score(query, answer, expected)
        rou = await rouge.score(query, answer, expected)
        jud = await judge.score(query, answer, expected)

        rows.append(
            {
                "id": entry["id"],
                "query": query,
                "context": entry["known_good_context"],
                "answer": answer,
                "expected": expected,
                "ours": {
                    "semantic": round(sem.score, 4),
                    "rouge": round(rou.score, 4),
                    "judge": round(jud.score, 4),
                },
            }
        )
    return rows


def _run_ragas(rows: list[dict]) -> dict[str, list[float]]:
    """Score the already-generated answers with Ragas. Returns a dict of
    metric_name -> per-row score (NaN where the small judge produced
    output Ragas could not parse, which is itself a finding)."""
    import math

    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import (
        answer_correctness,
        answer_similarity,
        context_precision,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    samples = [
        SingleTurnSample(
            user_input=r["query"],
            retrieved_contexts=[r["context"]],
            response=r["answer"],
            reference=r["expected"],
        )
        for r in rows
    ]
    dataset = EvaluationDataset(samples=samples)
    metrics = [answer_similarity, answer_correctness, faithfulness, context_precision]

    # max_workers=1: serialize the Ollama calls. The day-1 baseline found
    # that concurrent model work inflates latency through resource
    # contention; here it also keeps the local judge from being swamped.
    # timeout=900: Ragas's default per-job timeout is 180s, too tight for
    # a local 8B judge (the first call also cold-loads ~4.7GB into
    # memory). With the 3B judge the failure mode was unparseable JSON;
    # with the 8B judge the JSON parses but a too-tight timeout was
    # killing jobs, so the per-job budget is widened generously.
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=_build_ragas_llm(),
        embeddings=_build_ragas_embeddings(),
        run_config=RunConfig(max_workers=1, timeout=900),
        show_progress=True,
    )

    df = result.to_pandas()
    out: dict[str, list[float]] = {}
    for metric in metrics:
        col = metric.name
        out[col] = [
            None if (v is None or (isinstance(v, float) and math.isnan(v))) else round(float(v), 4)
            for v in df[col].tolist()
        ]
    return out


def _safe_mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(mean(present), 4) if present else None


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    entries = [
        e for e in load_yaml(GENERATION_SET)["entries"] if e.get("expected_answer") != "TODO"
    ]
    logger.info("scoring %d generation-set rows with our scorers + Ragas", len(entries))

    rows = await _generate_and_score_ours(entries)
    ragas_scores = _run_ragas(rows)

    # Attach Ragas scores per row and count parse failures (NaN).
    for i, r in enumerate(rows):
        r["ragas"] = {name: ragas_scores[name][i] for name in ragas_scores}

    ragas_means = {name: _safe_mean(vals) for name, vals in ragas_scores.items()}
    ragas_parse_failures = {
        name: sum(1 for v in vals if v is None) for name, vals in ragas_scores.items()
    }
    our_means = {
        scorer: round(mean([r["ours"][scorer] for r in rows]), 4)
        for scorer in ("semantic", "rouge", "judge")
    }

    report = {
        "captured": date.today().isoformat(),
        "model": MODEL,
        "ragas_judge_model": RAGAS_JUDGE_MODEL,
        "embed_model": EMBED_MODEL,
        "n_queries": len(rows),
        "query_set": "generation_set",
        "context_mode": "known_good (blessed), not live retrieval",
        "metric_map": RAGAS_TO_OURS,
        "our_scorer_means": our_means,
        "ragas_metric_means": ragas_means,
        "ragas_parse_failures": ragas_parse_failures,
        "per_query": rows,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("wrote %s", REPORT_PATH)
    print(json.dumps({k: report[k] for k in report if k != "per_query"}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
