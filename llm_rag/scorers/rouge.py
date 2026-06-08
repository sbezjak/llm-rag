from __future__ import annotations

from rouge_score import rouge_scorer

from llm_rag.scorers.base import Scorer, ScoreResult

DEFAULT_THRESHOLD = 0.40
DEFAULT_VARIANT = "rougeL"


class RougeScorer(Scorer):
    """Did the model's answer cover the words in the expected answer?

    ROUGE-L looks for the *longest common subsequence* (LCS) of words
    between the two texts. "Subsequence" means same order, but the
    matching words don't have to be next to each other, so "the cat sat
    on the mat" and "the cat quietly sat on the warm mat" share the LCS
    "the cat sat on the mat" (six words) even though they aren't
    identical strings. From that LCS length, ROUGE-L computes:

    - *recall*, how many of the *reference's* words made it into the
      LCS (did the model cover the reference?).
    - *precision*, how many of the *output's* words made it into the
      LCS (did the model stick to relevant words?).
    - *F1*, the harmonic mean of the two, what we report.

    Where BLEU emphasizes precision (translation cares: "did the model
    pick the right words?"), ROUGE emphasizes recall (summarization
    cares: "did the model cover the source?"). They are siblings, not
    competitors, both are vocabulary-overlap metrics that don't
    understand meaning, just from different angles.

    Backed by `rouge-score` (Google's reference implementation).
    Default variant is ROUGE-L F1, default threshold 0.40 follows
    summarization-paper convention.

    On RAG output (prose grounded in retrieved context) ROUGE-L is a
    blunt instrument: a correct answer phrased differently from the
    expected answer scores low. We keep it as a recall-flavored
    counterweight to the semantic scorer, not as a primary gate.
    """

    name = "rouge"

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        variant: str = DEFAULT_VARIANT,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold
        self.variant = variant
        self._scorer = rouge_scorer.RougeScorer([variant], use_stemmer=True)

    async def score(self, question: str, output: str, expected: str) -> ScoreResult:
        scores = self._scorer.score(expected, output)
        f1 = scores[self.variant].fmeasure
        passed = f1 >= self.threshold
        return ScoreResult(
            passed=passed,
            score=f1,
            reason=(
                f"{self.variant}_f1={f1:.3f} "
                f"{'>=' if passed else '<'} threshold={self.threshold}"
            ),
        )
