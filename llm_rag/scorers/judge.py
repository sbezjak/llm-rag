"""LLM-as-judge: a second model grades the first model's answer.

Plain-English version of what this file does, end to end:

1. Take the question, the expected answer, and the model's actual
   answer. Drop them into a grading-instructions template (see
   `RUBRIC_PROMPT`).
2. Send that filled-in prompt to an LLM (a *separate* call from
   whatever produced the answer being graded).
3. The LLM replies with a small JSON blob containing two integer
   scores (correctness 0-10, relevance 0-10) and a one-line
   explanation.
4. Parse the JSON, average the two scores, divide by 10 to get a
   score in [0, 1], compare against a threshold, return PASS or FAIL.

Why bother. The simpler scorers (ROUGE, semantic similarity) compare
*text shapes*. They can be fooled by:

- A right answer phrased very differently from the expected (ROUGE
  sees no overlap; embeddings see a different shape).
- A wrong answer that uses the right vocabulary (semantic similarity
  passes it).

A judge LLM reads both texts and can sometimes recognize "yes, this
answers the question" or "no, this is plausible-sounding nonsense"
where pure-text comparison can't.

Important caveat (pre-staked, PLAN.md decision 5). RAG answers are
longer and more context-laden than the short Q&A the eval harness
graded, and a small judge (llama3.2) struggles to grade them. When the
judge and the answering model are the same weights (the case here,
both llama3.2), the judge also has a blind spot for its own
hallucinations: it produced the same nonsense, so it can't tell it
apart from a real answer. This self-grading bias is a known weakness
to flag, not a finding to discover.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable

from llm_rag.providers.ollama import OllamaProvider
from llm_rag.scorers.base import Scorer, ScoreResult

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.7
DEFAULT_MODEL = "llama3.2"

# The grading instructions sent to the judge. See LLMJudgeScorer's
# docstring for why the rubric looks the way it does.
RUBRIC_PROMPT = """You are a strict but fair grader for a question-answering system.

You will be given:
- A QUESTION
- An EXPECTED answer (a reference, the model's answer does NOT need to
  match it word-for-word to be correct)
- The MODEL'S answer to grade

Grade the model's answer on two dimensions, each 0-10:
- correctness: does the answer correctly answer the question?
  10 = fully correct (even if phrased differently from the expected answer)
   5 = partially correct, or correct but with significant inaccuracies
   0 = wrong, contradicts the expected answer, or hallucinated
- relevance: is the answer on-topic and responsive to the question?
  10 = directly addresses the question
   5 = partially on-topic, drifts
   0 = off-topic or refuses to answer

Respond with ONLY a JSON object, no other text:
{{"reasoning": "<one or two sentences explaining your scores>",
 "correctness": <int>,
 "relevance": <int>}}

QUESTION: {question}
EXPECTED: {expected}
MODEL'S ANSWER: {output}"""


class JudgeParseError(RuntimeError):
    """Raised when the judge's response cannot be parsed into scores.

    A malformed judge response is an infrastructure problem (the judge
    model isn't following the rubric format), not "the model's answer
    was bad." Silently coercing to a 0 score would hide bugs in the
    judge prompt or the parsing logic.
    """


def _parse_judge_response(raw: str) -> tuple[int, int, str]:
    """Extract (correctness, relevance, reasoning) from the judge's text.

    Strategy: try strict JSON first. If the model wrapped the JSON in
    prose or markdown, try to find the first {...} block. As a last
    resort, regex-extract the two integer scores. If even that fails,
    raise JudgeParseError with the raw text, better than silently
    returning a 0.
    """
    try:
        data = json.loads(raw)
        return (
            int(data["correctness"]),
            int(data["relevance"]),
            str(data.get("reasoning", "")),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass

    json_blob = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_blob:
        try:
            data = json.loads(json_blob.group(0))
            return (
                int(data["correctness"]),
                int(data["relevance"]),
                str(data.get("reasoning", "")),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

    correctness_match = re.search(r'"?correctness"?\s*[:=]\s*(\d+)', raw, re.IGNORECASE)
    relevance_match = re.search(r'"?relevance"?\s*[:=]\s*(\d+)', raw, re.IGNORECASE)
    if correctness_match and relevance_match:
        return (
            int(correctness_match.group(1)),
            int(relevance_match.group(1)),
            "(reasoning unparseable, scores extracted by regex fallback)",
        )

    raise JudgeParseError(f"Could not parse judge response: {raw!r}")


class LLMJudgeScorer(Scorer):
    """Calls a grading LLM and turns its reply into a PASS/FAIL.

    Three rubric design choices are worth knowing about, since they
    explain why `RUBRIC_PROMPT` looks the way it does:

    - **Reference is a hint, not a string to match.** The judge is told
      the expected answer is a reference, the model's answer can be
      phrased completely differently and still be correct. This is what
      lets the judge pass right-but-different-shape answers.
    - **Two scores, not one.** Asking for *correctness* and *relevance*
      separately keeps two distinct failure modes from being smeared
      into a single number: a right-but-rambling answer scores high on
      correctness and low on relevance; a confident hallucination
      scores low on correctness and high on relevance. A single
      "quality" score loses that signal.
    - **Reasoning before score.** The rubric asks the judge to justify
      its score *before* writing the number down. A model asked for
      the score first tends to commit to a number and rationalize it.

    Construction.

    - `threshold`, pass/fail cutoff on the [0, 1] combined score.
    - `judge_fn`, async callable that takes a prompt and returns the
      judge's raw text. Defaults to a fresh `OllamaProvider` at
      temperature 0 (judge wants determinism). Tests inject a fake
      `judge_fn` to avoid touching HTTP at all.
    - `model`, the Ollama model name to use when `judge_fn` is the
      default.

    The rubric prompt and the judge's raw reply are logged at INFO so
    the pytest-html report shows exactly what the judge saw and said,
    even when a fake `judge_fn` is injected and no provider runs.
    """

    name = "judge"

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        judge_fn: Callable[[str], Awaitable[str]] | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold
        self.model = model
        if judge_fn is None:
            provider = OllamaProvider(model=model, timeout=180.0, temperature=0.0)
            judge_fn = provider.generate
        self._judge_fn = judge_fn

    def build_prompt(self, question: str, output: str, expected: str) -> str:
        """Render the rubric prompt the judge will see. Public so tests and
        debug paths can inspect what was actually sent without scoring."""
        return RUBRIC_PROMPT.format(question=question, expected=expected, output=output)

    async def score(self, question: str, output: str, expected: str) -> ScoreResult:
        prompt = self.build_prompt(question, output, expected)
        logger.info("judge rubric prompt (%s): %s", self.model, prompt)
        raw = await self._judge_fn(prompt)
        logger.info("judge raw response (%s): %s", self.model, raw)
        correctness, relevance, reasoning = _parse_judge_response(raw)

        # Clamp into 0-10 in case the judge invented an out-of-range score.
        correctness = max(0, min(10, correctness))
        relevance = max(0, min(10, relevance))

        score = (correctness + relevance) / 20.0
        passed = score >= self.threshold
        reason = (
            f"correctness={correctness}/10 relevance={relevance}/10 "
            f"score={score:.3f} {'>=' if passed else '<'} threshold={self.threshold} "
            f"| reasoning: {reasoning}"
        )
        return ScoreResult(passed=passed, score=score, reason=reason)
