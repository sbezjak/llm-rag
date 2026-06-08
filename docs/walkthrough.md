# Walkthrough

Live HTML test report: [report.html](https://sbezjak.github.io/llm-rag/reports/report.html)

I'm an automation tester. My job is checking that an app does the same thing every time. AI testing is the opposite: the same question can give a different answer every run, and more than one can be right.

RAG makes it harder. A wrong answer can come from two places: the search picked the wrong page, or it picked the right page and the model still got it wrong. From the outside they look the same. This repo is a pytest suite that tells them apart, with five test categories (retrieval, generation, refusal, citation, hallucination) over 12 pages of pytest documentation. The findings below are where the split paid off.

### Vocabulary

| In this repo | What a QA engineer would call it |
|---|---|
| Corpus | The only pages the system may answer from: 12 pytest doc pages. If the answer isn't here, it should refuse, not invent. |
| Chunk | One piece of a page (~1000 characters), with an id like `how-to__fixtures::factory-as-fixture::0`. The unit the search returns. |
| Embedding | Turns text into 384 numbers. Texts that mean the same thing get numbers that sit close together. |
| Vector retrieval | Finds chunks closest in meaning to the query. Good at paraphrase, bad at exact identifiers. |
| BM25 | Classic keyword search: counts shared words. The opposite of vector: good at identifiers, bad at paraphrase. |
| Hybrid (RRF) | Runs vector and BM25, then merges the two rankings. Catches what either misses alone. |
| Reranker | A slower second model that re-scores the top ~25 hits by reading query and chunk together. More precise, but can only reorder what it's given. |
| k / top-k | How many chunks the search hands back, like the first page of results. Tests use `k=5`; a test passes if the expected chunk is in that list. |
| Citation | The model writes `[chunk_id]` next to any claim it took from a chunk, so a test can check it used the chunk it pointed to. |
| `xfail(strict=True)` | A known limitation, allowed to fail. If it ever stops failing, the build breaks on purpose so the change can't slip by. |
| `mocked` / `chroma` / `ollama` markers | Three speed tiers: fake HTTP (~10s), real embeddings no model (~30s), live model (~20min). |
| p50 / p95 | Latency percentiles. p50 is the median, p95 is the slow tail (1 in 20). Same terms as any latency dashboard. |

### How it works

It's two services in a line: a **retriever** that finds the relevant snippets, and a **generator** (the `llama3.2` model) that reads those snippets and writes the answer. The bugs hide in the handoff between them.

This four-part retriever (vector + BM25 + RRF + cross-encoder reranker) is the standard production pattern.

Example of one question the whole way through:

```
query -> [ vector search ] + [ BM25 search ] -> fuse (RRF) -> rerank -> top 5 -> generator -> answer + citations
```

A user types **"how do I run only the tests marked slow?"** into the docs search.

1. **Two searches run at once** over the corpus, which has already been split into ~1000-character chunks (the model can't read whole pages at once, and search works better on small focused pieces) and indexed. Vector search turns the question into 384 numbers and finds chunks whose *meaning* is closest. BM25 ignores meaning and finds chunks that literally contain the words "marked", "slow", "tests".

2. **They each return their own ranked list,** and they often disagree. RRF merges the two lists into one, pushing up any chunk that ranked well in either, so a chunk both methods liked rises to the top. That merged list is the hybrid.
3. **The reranker re-scores the top ~25** of the hybrid list by reading the question and each chunk *together* (slower, but sharper than either search alone), and the best 5 are left.

4. **Those 5 chunks plus the question go to the generator.** The prompt tells the model to answer using only those chunks, to tag each claim with the id of the chunk it used, and to say "I don't know" if the chunks don't contain the answer.

5. **The answer comes back** with citations in square brackets like `[how-to__mark::select-by-marker::0]`. A regex pulls those ids back out so a test can check the model cited a chunk it was actually given.

Each test category checks one stop on this path: retrieval tests check steps 1-3 (did the right chunk reach the top 5?), generation and citation tests check steps 4-5 (good answer, honest citation?), and refusal tests check that the "I don't know" instruction in step 4 holds when the docs don't have the answer.

To see it happen, open the HTML report: every test row has a **"Captured log"** panel with the full prompt and response, untruncated, next to the assertion. You see what the model got and what it said, so you can tell a wrong chunk (retrieval) from a wrong answer (generation).

## TL;DR (5 minutes)

**What this is.** A pytest harness over a RAG pipeline (Chroma + `all-MiniLM-L6-v2` embeddings, BM25, `bge-reranker-base`, `llama3.2` generator) where retrieval and generation are tested as separate concerns. 12 findings, and 26 known failure modes pinned in the tests so they can't silently pass.

**What this isn't.** A production RAG framework. 12 pytest doc pages, 10 hand-written generation queries, 1 model. Findings are reproducible on this stack, not universal claims about RAG.

**The three biggest findings.**

1. **The model answers a question it should refuse** (F1). Ask how to write a pytest plugin, which isn't in the 12 pages, and instead of saying "I don't know" the model writes 2,000 characters built on an API that doesn't exist. It refuses 5 of 6 out-of-corpus questions but not this one: plugins are a real pytest topic it learned in training, so training overrides the instruction to refuse.

2. **The test that should catch a made-up source nearly passed it** (F2). For that same answer, the model cited a page that doesn't exist. The citation reader dropped the fake id and reported "cited nothing", the same signal a correct refusal gives, so the hallucination looked identical to good behaviour until a second reader was added to catch it.

3. **A search for a bare function name finds nothing** (F3). Query just `pytest.warns`, with the page that documents it in the index, and none of the four search methods find it. The user gets a related but wrong page and no signal anything went wrong. The fix isn't a better search engine, it's rewriting the query or indexing the docs differently.

### The five test categories

Five test categories, one for each way a RAG answer can be wrong. Each category isolates one of them.

| Test category | What goes wrong | What the user sees |
|---|---|---|
| Retrieval (`tests/retrieval/`) | The search never finds the right page | "It didn't tell me something that's in the docs." |
| Generation (`tests/generation/`) | Right page found, but the answer is still wrong | "It had the right page and got it wrong anyway." |
| Refusal (`tests/refusal/`) | Question isn't in the docs, model answers from training anyway | "It confidently answered something it shouldn't have." |
| Citation (`tests/citation/`) | The source it points to isn't the one it actually used | "The source link goes to the wrong page." |
| Hallucination (`tests/hallucination/`) | It points to a source that doesn't exist | "The source link is a dead 404." |

The same "wrong answer" can come from any of these. Without the split, you don't know which one to fix.

## Findings

Twelve findings, most surprising first. Each is a plain question, the test that asks it, and what it means.

### F1. The model answers a question it should refuse

**Question.** Ask the system something that isn't in the 12 pages, with a prompt that says to reply "I don't know based on the provided context." Does it refuse?

**Test.** Six out-of-corpus questions in `tests/refusal/`; `ooc-002` xfails strict.

```
ooc-002  query: how do I write my own pytest plugin and register a hook
  retrieval returns:   a topical-but-answer-free chunk (the plugin-LOADING
                       section, disabling assert rewriting, etc)
  model output:        ~2k chars of a confident answer using an invented API
                       (@pytest.addoption, a pytest_addoption method - neither
                       exists in real pytest)
  refused?             False (5/5 reruns, identical output every time)
```

**Takeaway.** It refused 5 of 6. The miss was "how do I write a pytest plugin", a real pytest topic the model knows from training. The instruction to use only the context is just more text in the prompt. It doesn't turn off what the model already learned. So on a topic it knows well, it answers anyway, and a related-but-answer-free chunk gave it an excuse. Refusing "what's the boiling point of water" is easy; refusing a real topic that just isn't in these pages is the hard case, and the dangerous one in production. The instruction to refuse won't hold on its own here. To catch it you'd add a separate check that the answer is actually backed by the retrieved chunks, and refuse when it isn't, instead of trusting the prompt. (The test uses the real retrieved chunk, not empty context, because that near-miss chunk is exactly what tempts the model to answer.)

### F2. The test that should catch a made-up source nearly passed it

**Question.** For that same plugin question, the model cited a page that doesn't exist in the corpus. Did the test catch the made-up source?

**Test.** `tests/hallucination/`, a pure-function test over the citation reader (no model call, so it pins the behaviour exactly).

```
ooc-002 output: "...you can use @pytest.addoption to register hooks
                 [how-to__plugin::registering-a-plugin::0]."

reads only real citations  -> []          (the fake id was filtered out
                                           because it isn't in the context)
reads everything           -> ["how-to__plugin::registering-a-plugin::0",
                               "-m", "my_option=foo"]
```

**Takeaway.** Almost not, and that is the finding. The model cited a page that doesn't exist. The citation reader was built to ignore any id that isn't a real chunk, so it dropped the fake one and reported "cited nothing". But "cited nothing" is also what the system reports when it correctly refuses. So a made-up citation and an honest refusal looked the same, and the test passed either way. The fix was a second reader that keeps every id the model wrote, real or fake, so a fabricated source shows up instead of disappearing. (Two smaller fixes fell out: filtering by the shape of a real id, since the regex also caught `["-m"]` from code examples; and moving the test onto the out-of-corpus questions, where fabricated sources actually appear.)

### F3. A search for a bare function name finds nothing

**Question.** A user types just `pytest.warns` into a docs search, and the page documenting `pytest.warns` is in the index. Does any search method find it?

**Test.** `ret-007`, run through all four retriever tests, all pinned as known failures.

```
ret-007  query: pytest.warns
  vector top-5:    MISS  (no sentence around it, so the embedding drifts to
                          topical neighbours - "warnings", "exceptions" - not the target)
  BM25 top-5:      MISS  (`pytest` is in every chunk, so it adds no signal;
                          `warns` alone isn't frequent enough to beat chunks
                          that match more words)
  hybrid top-5:    MISS  (merging two misses is still a miss)
  reranker top-5:  MISS  (can only reorder what hybrid passed it, and the
                          target chunk isn't in that set)
```

**Takeaway.** A bare identifier gives the search nothing to work with, and no later step can make up for it: keyword search can't outweigh how common `pytest` is, and the reranker can only reorder what it's handed. The fix isn't a better search, it's rewriting the query to add context, or indexing identifiers with more words around them.

### F4. The reranker usually finds the right page but ranks the wrong one first

**Question.** The reranker is the smart final step: it re-reads the question against each shortlisted chunk and reorders them. Does it put the *right* chunk at number 1?

**Test.** `tests/retrieval/test_reranker.py`, top-1 precision over 16 queries; the wrong chunk is ranked first on 10 of them, all pinned as known failures.

```
ret-013  query: --cache-clear
  reranker ranked #1:  cache::usage::0              (the cache page overview)
  right answer:        clearing-cache-content::0    (the chunk documenting --cache-clear)

ret-008  query: can I create a custom mark
  reranker ranked #1:  how-to-mark-test-functions-with-attributes::0   (the page intro)
  right answer:        registering-marks::0                            (the how-to subsection)
```

**Takeaway.** This is the largest cluster in the project, 10 of the 26 pinned. The right chunk almost always reaches the top 5 (recall is fine); the problem is ordering. It ranks broad page-overview or code-heavy chunks above the narrow subsection that actually answers the question. That's not a bug in this project's code, it's a bias of this particular off-the-shelf reranker (`bge-reranker-base`) on this corpus, and it matters anywhere the interface shows a single best answer. So if the UI shows only the top result, you can't lean on the reranker's #1 - show a few candidates, or fine-tune or swap the reranker for this kind of content.

### F5. Searching by meaning alone buries the right page just outside the top 5

**Question.** For a query like `pytest -m select tests by marker`, does meaning-search put the right page in the top 5 a user sees?

**Test.** `ret-014` and `ret-012`: the vector test xfails at top-5, the hybrid and reranker tests for the same rows pass.

```
ret-014  query: pytest -m select tests by marker
  vector top-5:    MISS  (returns chunks about *attaching* markers to
                          test functions instead of *selecting* tests
                          by marker - the chunk IS in vector top-10,
                          just buried)
  hybrid top-5:    HIT   (BM25 picks up `pytest -m` literally and
                          lifts it through RRF)
  reranker top-1:  HIT   (cross-encoder reading query + chunk
                          together puts the right one first)
```

**Takeaway.** Meaning-search left the right page just outside the top 5, even though it was sitting in the top 10. Keyword search caught the literal `pytest -m`, RRF lifted it into the top 5, and the reranker put it first. This is the whole argument for combining the searches: the queries users actually type, full of flags and function names, are the ones meaning-search ranks slightly too low, in the exact window the UI shows.

### F6. Leftover markup in the documents showed up in the answer

**Question.** The chunker is supposed to strip the docs' markup. Did any of it leak through into an answer?

**Test.** `gen-010`, generation quality.

```
gen-010  query: How do I set an environment variable for a single test?

The known-good context still contained the raw markup role:
    :py:meth:`monkeypatch.setenv <MonkeyPatch.setenv>`

The model copied it into the answer.
semantic 0.588 (fail), judge 7/10 (pass).
```

**Takeaway.** A chunk still held a raw markup role and the model copied it straight into the answer; stripping the missed markup raised the score from 0.588 to 0.684. This looked like a bad answer from the model, but the cause was a dirty document, and telling those two apart is the whole reason the categories are split.

### F7. Two scorers disagreed on the same answer

**Question.** Two scorers grade the same answer against the same reference. Do they agree?

**Test.** `gen-010`, scorer comparison in `reports/baseline_day1.json`.

```
gen-010  the model's answer:
  "You can use monkeypatch.setenv() to set an environment variable
   for a single test. [how-to__monkeypatch::...]"

  semantic:  0.722  (under the 0.75 gate - FAIL)
  judge:     7/10   (over the 0.7 threshold  - PASS)
```

**Takeaway.** The same answer failed one scorer and passed the other, and neither is wrong. It was correct but shorter than the reference. Similarity compares against the whole reference, so a shorter answer looks further away even when everything in it is right - that's the fail. The judge only asks whether it's a good answer, so it passed. They measure different things, which is why the harness runs several scorers but lets only one decide pass/fail. The 0.75 cutoff stayed put: a single point under the line isn't a reason to move it, because a threshold you lower to fit one answer stops testing anything.

### F8. The model cites its source in some sessions and not others

**Question.** The prompt tells the model to cite the chunk it used. At temperature 0, where it should be as repeatable as it gets, does it cite every time?

**Test.** `gen-002` and `gen-007`, citation compliance, pinned as non-strict known failures.

```
8 reruns of each row at temperature 0:
  gen-001  cited 8/8        (short answer, with a citation)
  gen-002  cited 0/8        (short "do X" answer, no citation)
  gen-005  cited 8/8
  gen-007  cited 0/8        (same kind of answer as gen-002)
  gen-010  cited 8/8
```

**Takeaway.** Some questions cite on all 8 reruns, some on 0. Within one session it's all 8 or none, never a mix. But across sessions the same question can flip. So temperature 0 doesn't mean the model always does the same thing, at least not for a small choice like whether to add a citation. That's why it's pinned as a non-strict failure: it won't break the build on the runs where it happens to pass. So if a citation has to appear every time, the prompt alone won't guarantee it - you'd add a check that catches an answer with no citation and asks the model again.

### F9. The reference answer has to match how long the model's answer is

**Question.** Two answers scored just under the bar. Should the reference answers be trimmed down to just what the question asks?

**Test.** `gen-001` and `gen-005`, generation quality.

**Takeaway.** Trimming helped one and hurt the other: gen-005 dropped from 0.857 to 0.696 while the judge still gave it 10/10. The similarity scorer is symmetric, so a reference that says less than the answer is just as far off as one that says more. There's no universal "trim to the question" rule; the reference has to match the length of the answer the model actually gives, which makes it a test fixture you tune against real output.

### F10. The same latency number was right or wrong depending on what else was running

**Question.** The first observability run timed retrieval at 35 to 189 seconds, slower than the LLM itself. Is that real?

**Test.** `run_day1_baseline.py`.

```
first run (everything running at once):  retrieval p50 34.6s,  p95 188.99s
timed on its own, before anything loads:  retrieval p50 1.67s,  p95   2.63s
```

**Takeaway.** No. Timed on its own, retrieval is about 2 seconds; the huge numbers came from timing it while the scorers and a second model were loading and running, all competing for the CPU. A latency number is only as good as how isolated the thing you're timing is, so the fix was to time retrieval in its own pass before anything else loads. The same trap shows up in production, where a fast call looks slow when the machine is busy with other work.

### F11. Run it twice and the answers are identical, only the timing moves

**Question.** Run the same 10 queries a second time, everything held constant. What moves?

**Test.** `run_drift_check.py` plus `tests/observability/test_drift.py`.

```
                         run 1      run 2      delta
tokens/query (mean)      460.8      460.8      +0.0%   (identical)
semantic / rouge / judge  same       same       0.0    (identical)
generation latency p95   10.85s     10.99s     +1.2%   (noise)
retrieval latency p95    2.63s      5.00s      +90%    (machine-load tail)
```

**Takeaway.** Every content number came back identical to the decimal - same tokens, same scores - and generation timing barely moved. Only the slowest retrieval runs jumped, enough to trip the regression flag, but that was the machine being busy again, not a real change. So at temperature 0 the content is stable. That makes a future content change a real signal worth looking into, while a jump in timing usually isn't. The drift check is tuned to match: it ignores big latency swings and flags small content changes.

### F12. These scorers match Ragas on the shared metric, but can't ask the two that need context

**Question.** Ragas is the standard library for evaluating a RAG pipeline. How do the three scorers in this repo compare to it?

**Test.** `run_ragas_comparison.py`, the same answers graded by both systems.

```
metric (Ragas)        this repo         Ragas    read
--------------------  ---------------   ------   -------------------------
answer_similarity     0.8324            0.8324   identical - same vectors,
                      (semantic)                 so the comparison is fair
answer_correctness    0.87              0.5062   far apart - Ragas grades
                      (judge)                    much harder
faithfulness          (none)            0.8929   context-aware, the local
                                                 scorers can't ask this
context_precision     (none)            1.0      =1.0 only because the test
                                                 feeds perfect context
```

**Takeaway.** The first row matched exactly, because both use the same vectors - that's the proof the comparison is fair. The next row split (0.51 vs 0.87): Ragas grades each claim separately, the judge here grades the whole answer at once, so Ragas is stricter. The real reason to reach for Ragas is the bottom two rows: they check whether the answer is backed by the source, which the scorers here can't do. The 1.0 only happened because the test feeds perfect context, so it isn't a real score for the retriever. That faithfulness row is the grounding check F1 called for - which is why Ragas is the tool you add once refusals and hallucinations matter, not just answer quality.

## Baseline numbers

These are the day-1 numbers later drift checks compare against. Captured 2026-05-27, llama3.2 at temperature 0, over the 10 generation queries with known-good context (so a retrieval miss can't move the quality numbers).

```
generation latency:  p50 4.19s,  p95 10.85s,  mean 5.21s
retrieval latency:   p50 1.67s,  p95  2.63s,  mean 1.84s
tokens per query:    460.8 total (386.2 prompt, 74.6 completion)
scorer means:        semantic 0.832,  rouge 0.320,  judge 0.870
```

The short read: generation is where the time goes, about 3x retrieval and the only part with a slow tail. Retrieval is cheap and steady, most of it the reranker. Semantic (0.83) and the judge (0.87) agree the answers are good; ROUGE (0.32) sits low because it counts word overlap against a short reference and the model paraphrases (F7 again).

## Known limitations

**1 corpus, 1 model.** 12 pages of pytest documentation, 16 retrieval queries, 10 generation queries, 6 out-of-corpus queries. Every finding is reproducible on this stack. Whether bare-identifier queries fail on a bigger embedding model over a 10k-chunk corpus is a question this repo doesn't answer. The *patterns* (split test categories, pinned known failures with a reason, latency timed in isolation) carry over; the specific numbers don't.

**`llama3.2` (3B) is both the model under test and the LLM judge.** The judge is the same model that wrote the answer - the classic self-grading risk. Softened by gating only on the semantic scorer (the judge is logged, not enforced) and by running Ragas with a stronger grader as a sanity check (F12). A real release gate would use a stronger judge that doesn't share weights with the system under test.

**Feeding known-good context inflates Ragas's context score to 1.0.** Generation runs on a known-good context fixture so a retrieval miss can't contaminate the quality numbers. The price is that context-quality metrics see an ideal retriever, not the real one. Reported with that caveat in F12, not treated as a retrieval win.

**Citation parsing is a regex over `[ ... ]`.** The production-grade version would have the model emit structured citations (JSON, span offsets, no string parsing). Deliberate trade-off: this keeps a single plain-text answer (no JSON mode, which `llama3.2` is unreliable at) and the citation stays visible to a human reader. The cost is the first half of F2 and the false citation from brackets in code - both documented, not hidden.

**Langfuse runs as a graceful no-op plus a local JSON tracer.** Standing up a real Langfuse server (a cloud account, or self-hosted Postgres and containers) is more than a portfolio repo needs. The traces are real and readable offline (`reports/baseline_day1_trace.json`); adding the keys and the optional extra ships them to a real server too.

**`n=10` for generation, `n=16` for retrieval.** Small enough that one moved query swings a percentage by 6 points, big enough to produce the findings above. More rows would tighten the numbers without teaching anything new at this stage.

## What I'd reuse

`pytest.xfail(strict=True, reason="<finding>")`. Every documented failure mode in this repo is pinned as a known failure with a reason string that names what's broken (a bare identifier has no signal for either search method; the similarity scorer is stricter than the judge on a short correct answer; the model leans on its training when the retrieved chunk is on a related topic but has no answer). 26 of them so far. They keep the suite green while documenting reality, and they live right next to the test that trips them. The payoff: the day a known failure starts passing, the build breaks on purpose, so someone updates the docs instead of losing the finding without noticing.

The technique isn't specific to AI. It works on any flaky integration where you understand *why* something fails. Use it whenever a test fails for a known structural reason you want recorded in code - and want to hear about it if the failure ever goes away.

It carried over from my first project ([llm-eval-harness](https://github.com/sbezjak/llm-eval-harness)), where it pinned scorer calibration against a 10-item set. What this project shows is that the same technique scales up to pinning 26 different retrieval, generation, refusal, and citation failures across a whole RAG pipeline without changing shape.

The HTML report writes to `reports/report.html` on every test run. The 26 pinned failures are the record of what the system is known to get wrong, so read those, not a coverage percentage.
