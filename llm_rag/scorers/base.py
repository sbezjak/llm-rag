from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreResult:
    """Outcome of a single scorer applied to a single (output, expected) pair.

    `score` is in [0.0, 1.0]. `passed` is the scorer's binary verdict, for
    threshold-based scorers (semantic, judge) it's `score >= threshold`.
    `reason` is a short human-readable explanation, surfaced in reports.
    """

    passed: bool
    score: float
    reason: str


class Scorer(ABC):
    """Scoring function over (question, output, expected).

    `score` is async because some scorers (LLM-as-judge) make HTTP calls at
    score time. Scorers without I/O (semantic similarity, ROUGE) just don't
    await anything. The suite runs under `asyncio_mode = "auto"`, so every
    test is async by default, async-only is the consistent choice given a
    single runtime context.

    Production frameworks (DeepEval, Ragas) expose dual sync+async
    interfaces to avoid async contagion across many runtime contexts.
    """

    name: str

    @abstractmethod
    async def score(self, question: str, output: str, expected: str) -> ScoreResult: ...
