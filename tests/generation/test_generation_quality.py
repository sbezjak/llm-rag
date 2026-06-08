"""End-to-end generation quality (requires live Ollama).

For each generation-set row we feed the model the hand-blessed
known-good context (NOT retrieval output), let it answer, and score the
answer three ways. Retrieval is bypassed, so a failure here is a
generation failure: right context in, bad answer out.

Rows whose `expected_answer` is still "TODO" are skipped, the scorers
need a reference to grade against. Fill the squishy part in
data/generation_set.yaml and these light up.

On the assertions: we hard-assert only SemanticScorer (does the answer
mean what a correct answer means?). ROUGE, the LLM judge, AND citation
compliance are logged but not gated. ROUGE is a blunt word-overlap
metric on free-form prose; llama3.2-as-judge is a known-weak grader for
long RAG answers (PLAN.md decision 5); and citation compliance is
non-deterministic at temperature 0 (notes.md finding 4: the model omits
the forced citation on some runs and emits it on others, same prompt and
context). Gating on any of the three would make this suite flaky for
reasons that are not generation correctness bugs in our code. Their
numbers still land in the pytest-html log for inspection. The
deterministic contract that our generator builds the citation-forcing
prompt and parses citations back out lives in the mocked test_generator.py.
"""

from __future__ import annotations

import logging

import pytest

from llm_rag.generators import generate_answer
from llm_rag.providers.ollama import OllamaProvider
from llm_rag.scorers import LLMJudgeScorer, RougeScorer, SemanticScorer
from tests.generation.conftest import chunks_for

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.ollama


@pytest.fixture(scope="module")
def provider() -> OllamaProvider:
    # Low temperature for answer stability across runs.
    return OllamaProvider(temperature=0.0)


@pytest.fixture(scope="module")
def semantic() -> SemanticScorer:
    return SemanticScorer()


@pytest.fixture(scope="module")
def rouge() -> RougeScorer:
    return RougeScorer()


@pytest.fixture(scope="module")
def judge() -> LLMJudgeScorer:
    return LLMJudgeScorer()


def _ids(entries: list[dict]) -> list[str]:
    return [e["id"] for e in entries]


@pytest.fixture
def gen_entry(request, generation_entries):
    return next(e for e in generation_entries if e["id"] == request.param)


# Rows whose semantic gate is a documented, reproducible failure rather than a
# regression. gen-010 (finding 3): the model's answer is correct but terse
# ("use monkeypatch.setenv() ..."), the LLM judge passes it, yet cosine sits at
# a rock-stable 0.722 (measured 8/8 runs, zero variance) just under the 0.75
# gate. strict=True so this XPASS-fails the moment the scorer disagreement
# closes, forcing us to re-examine the finding rather than letting it drift. The
# margin is thin (0.028), so a future across-session XPASS is a real signal to
# act on, not noise. Contrast citation compliance, which flips across sessions
# and is therefore logged, not xfailed (see module docstring + notes finding 4).
_XFAIL_SEMANTIC = {
    "gen-010": "finding 3: semantic scorer stricter than judge on a short correct answer",
}


def _param_for(gid: str):
    marks = []
    if gid in _XFAIL_SEMANTIC:
        marks.append(pytest.mark.xfail(reason=_XFAIL_SEMANTIC[gid], strict=True))
    return pytest.param(gid, marks=marks)


def pytest_generate_tests(metafunc):
    if "gen_entry" in metafunc.fixturenames:
        from tests.generation.conftest import GENERATION_SET
        from llm_rag.dataset import load_yaml

        entries = load_yaml(GENERATION_SET)["entries"]
        metafunc.parametrize("gen_entry", [_param_for(g) for g in _ids(entries)], indirect=True)


async def test_generation_quality(gen_entry, provider, semantic, rouge, judge):
    if gen_entry["expected_answer"] == "TODO":
        pytest.skip(f"{gen_entry['id']}: expected_answer not filled in yet")

    chunks = chunks_for(gen_entry)
    expected = gen_entry["expected_answer"]
    query = gen_entry["query"]

    result = await generate_answer(query, chunks, provider)

    sem = await semantic.score(query, result.answer, expected)
    rou = await rouge.score(query, result.answer, expected)
    jud = await judge.score(query, result.answer, expected)
    logger.info("scores %s: semantic=%s | rouge=%s | judge=%s", gen_entry["id"],
                sem.reason, rou.reason, jud.reason)

    assert result.answer.strip(), "model returned an empty answer"
    # Citation compliance: logged, NOT gated. At temperature 0 llama3.2 cites
    # its context chunk only intermittently (notes.md finding 4), so a gate
    # here would flake for a reason that is not a bug in our code. The
    # deterministic citation contract lives in the mocked test_generator.py.
    cited = chunks[0].id in result.cited_chunk_ids
    logger.info(
        "citation %s: model %s its context chunk (expected=%s cited=%s)",
        gen_entry["id"],
        "CITED" if cited else "did NOT cite",
        chunks[0].id,
        result.cited_chunk_ids,
    )
    # Quality gate: semantic only (see module docstring for why not ROUGE/judge/citation).
    assert sem.passed, f"semantic below threshold: {sem.reason}"
