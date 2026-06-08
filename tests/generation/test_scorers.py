"""Scorer behavior tests.

The judge test is fully mocked (a fake judge_fn, no HTTP). The semantic
and ROUGE tests run the real models but need no external service, so
they carry no marker and are skipped under `-m mocked`. They guard the
ported scorers against import/contract breakage, not answer quality.
"""

from __future__ import annotations

import pytest

from llm_rag.scorers import JudgeParseError, LLMJudgeScorer, RougeScorer, SemanticScorer
from tests.generation.conftest import fake_judge_fn


@pytest.mark.mocked
async def test_judge_parses_clean_json_and_passes():
    judge = LLMJudgeScorer(
        threshold=0.7,
        judge_fn=fake_judge_fn('{"reasoning": "good", "correctness": 9, "relevance": 8}'),
    )
    result = await judge.score("q", "model answer", "expected")
    assert result.passed
    assert result.score == pytest.approx((9 + 8) / 20.0)
    assert "correctness=9/10" in result.reason


@pytest.mark.mocked
async def test_judge_extracts_json_wrapped_in_prose():
    judge = LLMJudgeScorer(
        judge_fn=fake_judge_fn(
            'Sure! Here is my grade:\n{"correctness": 2, "relevance": 3}\nHope that helps.'
        ),
    )
    result = await judge.score("q", "bad answer", "expected")
    assert not result.passed
    assert result.score == pytest.approx((2 + 3) / 20.0)


@pytest.mark.mocked
async def test_judge_raises_on_unparseable_response():
    judge = LLMJudgeScorer(judge_fn=fake_judge_fn("I refuse to grade this."))
    with pytest.raises(JudgeParseError):
        await judge.score("q", "answer", "expected")


@pytest.mark.mocked
async def test_judge_build_prompt_contains_all_three_inputs():
    judge = LLMJudgeScorer(judge_fn=fake_judge_fn("{}"))
    prompt = judge.build_prompt("the question", "the output", "the expected")
    assert "the question" in prompt
    assert "the output" in prompt
    assert "the expected" in prompt


async def test_semantic_scorer_passes_identical_text():
    scorer = SemanticScorer()
    result = await scorer.score("q", "tmp_path gives a temp dir", "tmp_path gives a temp dir")
    assert result.passed
    assert result.score == pytest.approx(1.0, abs=1e-3)


async def test_rouge_scorer_passes_high_overlap():
    scorer = RougeScorer()
    expected = "use the tmp_path fixture to get a temporary directory"
    output = "you use the tmp_path fixture to get a temporary directory per test"
    result = await scorer.score("q", output, expected)
    assert result.passed
