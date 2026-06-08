"""Answer generation: prompt construction + provider call + citation parsing.

The generator is the seam between retrieval and the LLM. It takes the
ranked chunks retrieval produced, builds a prompt that pins each chunk
to its id and *forces the model to cite the chunk ids it used inline*,
calls the provider, then parses the cited ids back out of the answer.

Why force inline citation. The whole project exists to tell retrieval
bugs apart from generation bugs. If the model just emits prose, you
can't mechanically check whether it used the chunk it claimed to, or
whether it cited a chunk that was never retrieved. Forcing `[chunk_id]`
citations turns "did it use the right source?" into a string-set
assertion, which is exactly what the S5 citation and hallucinated-source
tests need.

Trade-off (written down per the repo's working style). Production RAG
usually returns *structured* citations (the model emits JSON, or the
framework tracks span offsets) rather than parsing ids out of free
text. We take the inline-bracket-and-regex approach instead because:

- it keeps the model call a single plain-text completion (no
  tool-calling / JSON-mode dependency on llama3.2, which is unreliable
  at strict JSON), and
- it makes the citation contract visible in the answer a reader sees,
  which suits a learning writeup.

The cost is parser fragility: a model that writes "chunk one" instead
of `[getting-started::...::0]` produces zero parsed citations. That
under-citation is itself a generation finding the S5 tests can catch,
so we surface it rather than hide it.
"""

from __future__ import annotations

import logging
import re
from typing import NamedTuple

from llm_rag.providers.base import Provider
from llm_rag.retrievers.vector import RetrievedChunk

logger = logging.getLogger(__name__)

# Matches one [...] citation. Chunk ids contain "::" and "-" but never
# "]", so "everything up to the closing bracket" is a safe capture.
_CITATION_RE = re.compile(r"\[([^\]]+)\]")

PROMPT_TEMPLATE = """You answer questions about pytest using ONLY the context below.

Each context chunk is labelled with an id in the form [chunk-id]. Rules:
- Answer using only facts found in the context. Do not use outside knowledge.
- After each claim, cite the id(s) of the chunk(s) it came from, inline,
  in square brackets exactly as shown, for example: [{example_id}].
- If the context does not contain the answer, reply exactly:
  I don't know based on the provided context.
  and cite nothing.

Context:
{context}

Question: {query}

Answer (with inline [chunk-id] citations):"""


class GeneratedAnswer(NamedTuple):
    """Result of a generation call.

    A NamedTuple so callers can unpack it as `(answer, cited)` per the
    original contract, while S5 tests get named access to
    `.cited_chunk_ids`.
    """

    answer: str
    cited_chunk_ids: list[str]


def build_prompt(query: str, ranked_chunks: list[RetrievedChunk]) -> str:
    """Render the generation prompt. Public so tests can inspect it."""
    context = "\n\n".join(f"[{c.id}]\n{c.text}" for c in ranked_chunks)
    example_id = ranked_chunks[0].id if ranked_chunks else "chunk-id"
    return PROMPT_TEMPLATE.format(context=context, query=query, example_id=example_id)


def parse_all_citations(answer: str) -> list[str]:
    """Pull every bracketed id out of the answer, UNFILTERED.

    Unlike `parse_citations`, this does not drop ids that were absent
    from the context: it returns exactly what the model wrote in
    brackets, de-duplicated in order of appearance. That is the input
    the hallucinated-source tests need, an id the model cited that is
    not in the index is invisible to `parse_citations` (it filters such
    ids out by design), so detecting a hallucinated citation requires
    comparing the raw bracket contents against the index here.
    """
    seen: list[str] = []
    for match in _CITATION_RE.findall(answer):
        for raw in re.split(r"[,\s]+", match.strip()):
            cid = raw.strip()
            if cid and cid not in seen:
                seen.append(cid)
    return seen


def parse_citations(answer: str, known_ids: set[str]) -> list[str]:
    """Pull cited chunk ids out of the answer text.

    Only ids that were actually in the context (`known_ids`) are kept,
    in order of first appearance, de-duplicated. An id the model cited
    that is NOT in `known_ids` is dropped here but is the signal the
    hallucinated-source tests assert on, so callers that care should
    compare the raw bracket contents against the index themselves.
    """
    seen: list[str] = []
    for match in _CITATION_RE.findall(answer):
        # A single [a, b] citation may list several ids.
        for raw in re.split(r"[,\s]+", match.strip()):
            cid = raw.strip()
            if cid in known_ids and cid not in seen:
                seen.append(cid)
    return seen


async def generate_answer(
    query: str,
    ranked_chunks: list[RetrievedChunk],
    provider: Provider,
) -> GeneratedAnswer:
    """Build the prompt, call the provider, return (answer, cited ids).

    Pure over its inputs apart from the single provider call: retrieval
    has already happened, so this function never touches Chroma or BM25.
    Tests inject known-good `ranked_chunks` as a fixture and a mocked
    provider, keeping generation failures from contaminating retrieval.
    """
    prompt = build_prompt(query, ranked_chunks)
    answer = await provider.generate(prompt)
    known_ids = {c.id for c in ranked_chunks}
    cited = parse_citations(answer, known_ids)

    logger.info("generate_answer query=%r chunks=%d cited=%s", query, len(ranked_chunks), cited)
    return GeneratedAnswer(answer=answer, cited_chunk_ids=cited)
