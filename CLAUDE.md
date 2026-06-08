# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Pytest-based RAG (retrieval-augmented generation) system plus its own
test harness. Targets Ollama (`llama3.2` via `localhost:11434`) as the
LLM backend and ChromaDB (embedded, file-backed) as the vector store.
Python 3.11+, managed with `uv`.

**One focused theme:** how do you test a RAG pipeline so retrieval
bugs (wrong chunk fetched) and generation bugs (right chunk, bad
answer) can be told apart? See `PLAN.md` for full scope discipline.

Package layout under `llm_rag/`:

- `providers/`, adapters that talk to LLM backends (only
  `OllamaProvider`). Provider code is the only place that issues HTTP
  calls; tests mock at this boundary with `respx`. Shape ported from
  llm-eval-harness.
- `retrievers/`, chunk retrieval logic. Vector (Chroma), BM25, and a
  reciprocal-rank-fusion hybrid. A `bge-reranker-base` cross-encoder
  sits on top of the hybrid results. Retrievers are pure functions over
  `(query, index) -> ranked_chunks`.
- `generators/`, prompt construction + provider call. Forces inline
  citation of chunk ids so citation-correctness is testable.
- `scorers/`, copied in shape from llm-eval-harness:
  `SemanticScorer` (sentence-transformers, cosine, `all-MiniLM-L6-v2`),
  `RougeScorer`, `LLMJudgeScorer` (hybrid rubric, second Ollama call).
  Same async `Scorer.score(question, output, expected) -> ScoreResult`
  contract.
- `observability/`, Langfuse tracing wrappers around retrievers and
  generators. Captures latency / token / scorer numbers for the day-1
  baseline.
- `dataset.py`, loads + validates the YAML datasets under `data/`.

Datasets live at `data/`:

- `corpus/`, 12 pytest documentation pages fetched as reStructuredText (`.rst`) from the pytest GitHub repo. The chunker strips rst directives and splits on section headers.
- `retrieval_set.yaml`, hand-written `(query, expected_chunk_id)` pairs.
- `generation_set.yaml`, `(query, known_good_context, expected_answer)`.
- `out_of_corpus.yaml`, questions whose answers are not in the corpus,
  used to test "I don't know" handling.

Do not relocate the package layout, `pyproject.toml` pins
`packages = ["llm_rag"]` for the wheel build.

## Commands

Use `uv` for all environment and execution tasks:

- Install / sync deps (including dev group): `uv sync`
- Run all tests: `uv run pytest`
- Run a single test: `uv run pytest tests/path/to/test_file.py::test_name`
- Run only fast (mocked) tests: `uv run pytest -m mocked`
- Run only tests that hit a real Ollama: `uv run pytest -m ollama`
- Run only tests that touch Chroma: `uv run pytest -m chroma`
- Skip Ollama tests: `uv run pytest -m "not ollama"`
- HTML report: auto-generated at `reports/report.html` on every `uv run pytest` (self-contained, includes captured logs at INFO+).
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`

Project-specific scripts (added incrementally across sessions):

- Fetch corpus: `uv run python -m llm_rag.scripts.fetch_corpus`
- Build / rebuild the Chroma index: `uv run python -m llm_rag.scripts.build_index`
- Dump human-readable chunk manifest: `uv run python -m llm_rag.scripts.dump_manifest` (writes `data/chunk_manifest.md`)
- Capture retrieval baseline (recall@k per retriever): `uv run python -m llm_rag.scripts.run_baseline` (writes `reports/retrieval_baseline.json`)
- Capture day-1 observability baseline (latency p50/p95, tokens per query, scorer means): `uv run python -m llm_rag.scripts.run_day1_baseline` (writes `reports/baseline_day1.json`, needs live Ollama + built index)
- Drift check (re-run the same queries, diff against the day-1 baseline): `uv run python -m llm_rag.scripts.run_drift_check` (writes `reports/drift_report.json` + a kept `reports/baseline_<date>.json` snapshot, exits non-zero on a flagged regression, needs live Ollama + built index)
- Ragas comparison (S8 stretch, our 3 scorers vs Ragas metrics on the same answers): `uv sync --extra ragas` then `uv run python -m llm_rag.scripts.run_ragas_comparison` (writes `reports/ragas_comparison.json`, needs live Ollama + built index; Ragas is an optional extra, not a core dep)

## Test conventions (configured in pyproject.toml)

- `asyncio_mode = "auto"`, async tests do not need `@pytest.mark.asyncio`.
- Three custom markers gate environment-dependent tests:
  - `@pytest.mark.ollama`, slow, requires a live Ollama instance.
  - `@pytest.mark.chroma`, requires a built Chroma index (or a per-test
    fixture-built one in `tmp_path`).
  - `@pytest.mark.mocked`, uses `respx` to mock the Ollama HTTP API;
    should be the default for unit tests.
- `testpaths = ["tests"]`; `ruff` line length is 100, target `py311`.

Test categories (mirrored in `tests/` subfolders):

- `tests/retrieval/`, assert the expected chunk id (or one of an
  expected set) appears in top-k for a given query. Failing here means
  embedding / chunking / BM25 / RRF / reranker is wrong.
- `tests/generation/`, given known-good context, score the final answer
  with the three scorers.
- `tests/refusal/`, out-of-corpus questions; assert the system refuses
  or says it doesn't know.
- `tests/citation/`, the cited chunk id is actually the one used to
  produce the answer.
- `tests/hallucination/`, the system never cites a chunk id that is
  not in the index.

Most RAG bugs are retrieval bugs masquerading as generation bugs,
keeping the categories separate is how you tell them apart.

## Architecture intent

The package layout signals the intended seams; preserve them when
adding code:

- HTTP calls only inside `providers/`. Tests mock here with `respx`.
- Retrieval has no LLM calls. It's deterministic given the index and
  the query, so retrieval tests don't need the `ollama` marker.
- Generation depends on retrieval output but tests inject known-good
  context as a fixture, so retrieval and generation failures don't
  contaminate each other.
- Scorers stay I/O-free (`LLMJudgeScorer` is the exception, it calls a
  provider).
- Observability is decoration on top of retrievers + generators, never
  inside them, so it can be turned off in tests without changing
  behavior.

`httpx` is the transport used by `OllamaProvider` and `respx` mocks.
ChromaDB calls are real but tests marked `chroma` use a per-session
collection in `tmp_path`.

## Working style with this user

- **Prepare drafts/templates for any task the user has to do by hand.**
  When the next step is something only the user can do (write a
  retrieval-set row, decide which chunk is the "expected" one, grade an
  out-of-corpus refusal), prepare a fill-in-the-blanks file with the
  structure pre-built. Don't make the user start from a blank page.
  Examples: a YAML scaffold with TODOs, a markdown table with rows
  pre-filled from the index, a notes template with section headings.
  Reduce the user's task to filling in the squishy parts.
- **Capture explanations to `notes.md` when teaching.** When the user
  asks "explain this to me" and the answer is non-trivial (chunking
  trade-offs, RRF math, reranker mechanics, why hybrid beats pure
  vector), mirror it (lightly cleaned up) into `notes.md` as
  reference material. The chat scrolls; the article stays.
- **Two writing registers, no duplicate copies.** `notes.md` is the
  dense, finding-first record. The teaching / front-door artifacts (the
  narrated walkthrough, `docs/` explainers, README) use the plain,
  layered voice the user likes (plain words, an analogy or two,
  structured so a reader can stop as soon as they have it). Same
  explanation in two registers drifts out of sync, so the registers
  serve different artifacts, never two copies of one thing. Do not add a
  standalone concepts/glossary file; notes.md plus the walkthrough cover
  both lookup and teaching.
- **Default to production / best-practice solutions; take the pragmatic
  shortcut only when the trade-off is justified for this project's
  scope, and call the trade-off out explicitly.** Don't silently pick
  the easy path. Name what the production-grade pattern would be, name
  why we're not doing it here (scope, runtime context, learning
  focus), and write the trade-off into `notes.md` so the writeup
  shows it was a deliberate choice, not an oversight.
- **Prefer the smallest solution that solves the problem; after writing,
  re-read and cut machinery the task didn't ask for.** Asked to "add
  validation", don't reach for a configurable framework with custom
  exception classes when a few lines of `if` are the real answer. The
  extra structure is plausible-looking but not load-bearing - it just
  becomes more to read, test, and maintain. Treat the re-read as a step:
  "is half of this scaffolding I invented but nobody asked for?" If yes,
  rewrite it small. This is the same working-with-AI finding the project
  is about (over-engineered code is exactly the plausible-looking
  garbage that isn't load-bearing), not generic advice.
- **Validate a new integration cheaply before any long or expensive
  run.** Before launching a long run (especially one exercising a new
  external tool that has never run end to end here), prove it works on
  the smallest possible input first (1-2 items, fabricated inputs are
  fine) and confirm the output parses / scores. ALWAYS arm a monitor on
  the log for any run that is not near-instant - not only "inherently
  long" ones - grepping for success AND failure signatures
  (`500|Internal Server Error|Traceback|Error|Exception|Killed|OOM|NaN|Timeout|Failed`)
  so a mid-run failure surfaces at the failing job, not when the user
  notices nothing finished and kills it by hand. Smoke test and monitor
  are two SEPARATE obligations: a passing smoke test only proves item 1
  works and is never a reason to launch the full run unmonitored, the
  full run can still 500 / OOM / hang on a later item. Smoke test first,
  monitor second, both every time. When a run does fail, fix the root
  cause and re-validate cheaply before re-running full. (Learned in S8: a
  blind Ragas run burned ~35 minutes before it was clear llama3.2 could
  not satisfy Ragas's JSON-schema contract; later runs revealed a
  too-tight timeout and a Mac GPU out-of-memory clash; and a later run
  passed its smoke test but was launched unmonitored, hit a 500 partway,
  and wasted ~5 min before the user caught it - each a distinct miss a
  smoke test plus a monitor would have caught.)
- **Always make model prompts and responses visible in test reports.**
  Every component that calls a model (providers, generators, LLM-judge
  scorers) must log the prompt going in and the response coming out at
  `INFO` level via the stdlib `logging` module, not truncated. The
  always-on pytest-html report captures these, so any failing test's
  "Captured log" section shows exactly what the model saw and said.
  This is non-negotiable for a RAG project where the whole point is
  telling retrieval bugs apart from generation bugs.
- In all prose (docs, comments, commit messages, PR descriptions), join clauses
    with a single hyphen `-`, a comma, a period, or parentheses. The only dash
    character in written text is a single `-`.
- End commit messages at the body. The user is the sole author.
