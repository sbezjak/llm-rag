# llm-rag

> Part of a [5-project AI/QA testing portfolio](https://github.com/sbezjak/sbezjak) - all projects and write-ups.

New here? The [walkthrough](docs/walkthrough.md) is the guided tour - why
the project exists, every finding with evidence. This README is the
reference.

Live report: [test report](https://sbezjak.github.io/llm-rag/reports/report.html)
Most RAG failures look like generation bugs but are actually retrieval bugs -
the model produced a wrong answer because the wrong chunk was fetched, not
because the model failed at reading. This project is a pytest-based RAG
system plus its own test harness that treats retrieval and generation as
separately-testable concerns, so the two failure modes can be told apart.

The three project-local scorers (`SemanticScorer`, `RougeScorer`,
`LLMJudgeScorer`) are also compared against [Ragas](https://docs.ragas.io)
on the same answers, so the writeup can say concretely what a
purpose-built RAG eval library measures that our scorers structurally
cannot, and where it just overlaps (see Finding 9).

The system targets [Ollama](https://ollama.com/) (`llama3.2` at
`localhost:11434`) as the LLM backend and
[ChromaDB](https://www.trychroma.com/) as the vector store. Python 3.11+,
managed with [`uv`](https://docs.astral.sh/uv/).

## Quickstart

```bash
uv sync                                # install runtime + dev deps
ollama pull llama3.2                   # ~2 GB, one-time (only for live tests)

uv run python -m llm_rag.scripts.fetch_corpus    # one-time: 12 pytest doc pages
uv run python -m llm_rag.scripts.build_index     # one-time: chunk + embed into Chroma

uv run pytest -m mocked                # fast path, no model, no network (~10 s)
uv run pytest -m "not ollama"          # mocked + chroma (~30 s)
uv run pytest                          # full suite, needs live Ollama + index

uv run ruff check .                    # lint
uv run ruff format .                   # format
```

HTML test report is auto-generated at `reports/report.html` on every
`uv run pytest`.

## Findings

Eleven findings from building and testing this system. The right column
is the code entry point where a finding lives in a test; a few live only
in the walkthrough, which carries the full evidence for each
([`docs/walkthrough.md`](docs/walkthrough.md)).

| # | Finding | Where |
|---|---|---|
| 1 | A bare-identifier query (`pytest.warns`) can fail across every retriever - vector, BM25, hybrid, reranker - because the embedding has too little sentence context and BM25 fragments the identifier into low-signal tokens | `tests/retrieval/test_vector.py`, `test_bm25.py`, `test_hybrid.py`, `test_reranker.py` (xfailed strict) |
| 2 | Pure vector retrieval can bury the right chunk just outside top-5 even when hybrid + reranker recover it - the reason production RAG is not pure vector | `tests/retrieval/test_vector.py` (xfailed strict on ret-012 / ret-014) |
| 3 | rst markup leaks from chunks into generated answers | `tests/generation/test_generation_quality.py` |
| 4 | Semantic and LLM-judge scorers disagree on the same answers | `tests/generation/test_scorers.py` |
| 5 | Citation compliance is non-deterministic at temperature 0, and the variance is session-level (not per-call) | `tests/citation/test_citation.py` |
| 6 | Reference-answer length must match the model's actual answer scope, or the scoring threshold drifts | walkthrough |
| 7 | Refusal holds 5 of 6 out-of-corpus queries; breaks on plugin authoring | `tests/refusal/test_refusal.py` |
| 8 | `parse_citations` silently dropped fabricated sources; a separate `parse_all_citations` was needed for the hallucination tests | `tests/hallucination/test_hallucination.py` |
| 9 | Where a measurement is placed decides what it measures: retrieval latency was inflated by concurrent model load until measured in isolation | walkthrough + `run_day1_baseline.py` |
| 10 | At temperature 0, content is reproducible across reruns; only latency wanders | `tests/observability/test_drift.py` |
| 11 | Ragas overlaps one of our scorers exactly, disagrees with another (the grader is the variable), and adds faithfulness (the real gap) | walkthrough + `run_ragas_comparison` |

Findings 1-8 are retrieval / answer / citation behaviors caught by
the test suite. Findings 9-11 come from the observability and
scorer-comparison work.

## Overview

The project answers one question: *how do you test a RAG pipeline so you
can tell retrieval bugs apart from generation bugs?* A wrong answer can
come from at least five places (wrong chunk fetched, right chunk ignored,
right chunk misread, wrong chunk id cited, or a refusal that should have
fired). They look identical from the outside; each needs a different fix.
The pytest layout is the answer: five test categories, each isolating one
failure mode.

| Category | Question | Where |
|---|---|---|
| Retrieval | Does the right chunk appear in top-k? | `tests/retrieval/` |
| Generation | Given known-good context, is the answer correct? | `tests/generation/` |
| Refusal | Does the system say "I don't know" when the answer isn't in the corpus? | `tests/refusal/` |
| Citation | Does the cited chunk id actually match the one used? | `tests/citation/` |
| Hallucination | Does the system ever cite a chunk id that isn't in the index? | `tests/hallucination/` |

Each retriever and scorer is small, focused, and unit-tested at its
boundary:

| Component | Backend | Purpose |
|---|---|---|
| `vector_retrieve` | Chroma + `all-MiniLM-L6-v2` embeddings | Dense semantic retrieval |
| `bm25_retrieve` | `rank-bm25` | Sparse lexical retrieval |
| `hybrid_retrieve` | Reciprocal rank fusion over vector + BM25 | Production-realistic hybrid |
| `Reranker` | `BAAI/bge-reranker-base` cross-encoder | Reorder hybrid candidates |
| `SemanticScorer` | `sentence-transformers` cosine similarity | Answer ≈ expected by meaning |
| `RougeScorer` | `rouge-score` ROUGE-L F1 | Lexical overlap |
| `LLMJudgeScorer` | Second Ollama call, hybrid correctness/relevance rubric | LLM grades LLM |

## Testing

The suite is split by three custom markers (defined in `pyproject.toml`):

| Marker | Purpose | Runtime |
|---|---|---|
| `mocked` | `respx`-mocked Ollama; pure unit tests on retrievers, scorers, prompt build | ~10 s |
| `chroma` | requires a built Chroma index (real embeddings, no LLM) | ~30 s |
| `ollama` | live model + judge calls | ~20 min on a Mac (M-series, llama3.2 loaded) |

`asyncio_mode = "auto"` is set, so async tests do not need
`@pytest.mark.asyncio`.

Every component that calls the model (providers, generators, the LLM-judge
scorer) logs the full prompt and the full response at `INFO`. The
always-on pytest-html report captures these, so any failing test's
"Captured log" section shows exactly what the model saw and said.

### `xfail(strict=True)` as executable documentation

Known limitations are encoded as `xfail(strict=True)` tests with a `reason`
that names the finding. Example, the reranker's top-1 weaknesses on this
corpus:

```
test_reranker.py::test_reranker_top1_precision[ret-001]  XFAIL
  reason: bge picks parametrize::basic-pytest-generate-tests-example::0
  over getting-started::create-your-first-test. Reranker buries
  plain-English walkthrough chunks under identifier-heavy ones.
```

If a known failure ever silently *stops* failing, strict mode flips the
test red and forces investigation. The xfail set is the spec for what the
system is known to get wrong.

## Scripts

Each runs as `uv run python -m llm_rag.scripts.<name>`:

| Script | Purpose | Needs |
|---|---|---|
| `fetch_corpus` | Download 12 pytest doc pages into `data/corpus/` | network |
| `build_index` | Chunk the corpus + embed + persist a Chroma collection | corpus on disk |
| `dump_manifest` | Write a human-readable chunk index to `data/chunk_manifest.md` | built index |
| `run_baseline` | Capture retrieval recall@k per retriever to `reports/retrieval_baseline.json` | built index |
| `run_day1_baseline` | Capture day-1 observability baseline (latency p50/p95, tokens, scorer means) | live Ollama + built index |
| `run_drift_check` | Re-run the baseline and diff against day-1; non-zero exit on regression | live Ollama + built index + a day-1 baseline |
| `run_ragas_comparison` | Compare our 3 scorers vs Ragas's 4 metrics. Install with `uv sync --extra ragas` and `ollama pull llama3.1:8b` (Ragas's grader needs the larger model) | live Ollama + built index + ragas extra |

## Observability

Tracing wraps retrieval and generation from the outside. Spans carry
latency, token counts, returned chunk ids, and prompt/answer sizes. Two
backends behind one interface:

- **Local JSON tracer** (default). Spans collected in memory, flushed to
  `reports/baseline_*_trace.json`. No network, no account, no extra deps.
- **Langfuse tracer**. If `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` are
  set and the `observability` extra is installed (`uv sync --extra
  observability`), spans are additionally shipped to a Langfuse server.
  The pipeline never breaks if Langfuse is absent; it falls back to the
  local tracer and logs that it did.

The day-1 baseline (`run_day1_baseline`) captures a fixed point that
future runs diff against via `run_drift_check`. The drift script is
tolerance-aware: latency wanders run-to-run even at temperature 0, so a
50% latency swing is not flagged, but a 10% token shift or a 5% scorer
drop is, and any query that was passing the quality floor and now isn't
is reported per-row.

## Layout

```
llm_rag/
├── providers/        # backend adapters; only place that issues HTTP
├── retrievers/       # vector, bm25, hybrid (RRF), reranker
├── generators/       # prompt build + provider call + citation parsing
├── scorers/          # semantic, rouge, llm-judge
├── observability/    # tracing + drift; pure decoration
├── scripts/          # CLI entry points (see Scripts)
└── dataset.py        # YAML loader
data/
├── corpus/                  # 12 pytest doc pages (.rst)
├── retrieval_set.yaml       # (query, expected_chunk_id) pairs
├── generation_set.yaml      # (query, known_good_context, expected_answer)
├── out_of_corpus.yaml       # questions not answerable from the corpus
└── chunk_manifest.md        # human-readable chunk index
reports/                     # baselines, drift reports, html test report
docs/walkthrough.md          # layered writeup
```

## Further reading

- [`docs/walkthrough.md`](docs/walkthrough.md), layered writeup: why this
  project exists, TL;DR, every finding with evidence, known limitations.
- [`CLAUDE.md`](CLAUDE.md), guidance for AI assistants working in this repo.

The canonical sources behind this project, worth reading first for the
overview a hands-on build does not give:

- Lewis et al. 2020, [Retrieval-Augmented Generation for Knowledge-Intensive
  NLP Tasks](https://arxiv.org/abs/2005.11401), the paper that introduced RAG,
  the parametric + non-parametric memory split this project tests.
- Es et al. 2023, [Ragas: Automated Evaluation of Retrieval Augmented
  Generation](https://arxiv.org/abs/2309.15217), the paper behind the
  [Ragas](https://docs.ragas.io) library compared in Finding 11. It defines
  faithfulness, answer relevance, and context precision, the
  retrieval-vs-generation metric split this suite mirrors.
- The retrieval primitives' origins: BM25 (Robertson & Zaragoza, 2009, *The
  Probabilistic Relevance Framework: BM25 and Beyond*) and Reciprocal Rank
  Fusion (Cormack et al., 2009), the sparse retriever and the hybrid fusion
  used here.
