from __future__ import annotations

from sentence_transformers import SentenceTransformer, util

from llm_rag.scorers.base import Scorer, ScoreResult

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.75


class SemanticScorer(Scorer):
    """Does the model's answer *mean* the same thing as the expected answer?

    1. A small neural network (the *embedding model*) turns each piece of
       text into a list of 384 numbers, a coordinate in a 384-dimensional
       "meaning space."
    2. Texts with similar meanings end up at nearby coordinates because the
       model was trained that way. "Paris" and "The capital of France is
       Paris." sit close; "Paris" and "Water boils at 100C" sit far.
    3. We measure how close the two coordinates are with *cosine
       similarity* (the angle between them, not the distance, same
       direction = 1.0, perpendicular = 0.0).
    4. If the similarity is above a threshold, the scorer says PASS.

    None of this is a real understanding of meaning, it is pattern
    matching over how words co-occur in the model's training data.

    Two implementation details:

    - The default model is `all-MiniLM-L6-v2`: small, fast, downloads
      automatically the first time the scorer is constructed. Same model
      as the eval harness, so semantic scores are comparable across
      projects.
    - The threshold is the knob you tune against your own dataset.
      Too high collapses into exact match. Too low lets unrelated
      answers through. 0.75 is a starting point, expect to revisit
      it once you have real data.
    """

    name = "semantic"

    # Class-level cache: loading a model takes seconds and downloads weights
    # on first use. One process, one model load per model name, regardless
    # of how many SemanticScorer instances are created.
    _model_cache: dict[str, SentenceTransformer] = {}

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold
        self.model_name = model_name
        if model_name not in self._model_cache:
            self._model_cache[model_name] = SentenceTransformer(model_name)
        self._model = self._model_cache[model_name]

    async def score(self, question: str, output: str, expected: str) -> ScoreResult:
        embeddings = self._model.encode([output, expected], convert_to_tensor=True)
        cosine = util.cos_sim(embeddings[0], embeddings[1]).item()
        clamped = max(0.0, min(1.0, cosine))
        passed = clamped >= self.threshold
        return ScoreResult(
            passed=passed,
            score=clamped,
            reason=f"cosine={clamped:.3f} {'>=' if passed else '<'} threshold={self.threshold}",
        )
