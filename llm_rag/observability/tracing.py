"""Tracing decoration over retrievers and generators.

Architecture rule (CLAUDE.md): observability is decoration on top of
retrieval and generation, never inside them. So nothing in
`retrievers/` or `generators/` imports this module. Callers (the day-1
baseline script, and any test that wants traces) wrap the pure calls
here, and turning tracing off is just not wrapping, behaviour is
unchanged either way.

Two backends behind one interface:

- `LocalTracer` collects spans in memory and writes them to a JSON
  trace file. No network, no account. This is the default and the
  honest-scope choice for a local-only Ollama project (trade-off
  written up in notes.md): the spans are inspectable offline and feed
  the day-1 baseline directly.
- `LangfuseTracer` additionally ships each span to a Langfuse server
  when credentials are present (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY
  in the environment) and the SDK is installed. If either is missing we
  fall back to `LocalTracer` and say so at INFO, so the pipeline never
  breaks for lack of a Langfuse instance.

`get_tracer()` picks the backend. The span API is identical for both,
so the baseline script does not care which one it got.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """One timed unit of work (a retrieval or a generation call).

    `attrs` carries whatever the caller wants on the trace: the query,
    returned chunk ids, token counts, the answer. `set` is a small
    convenience so the caller can record outputs from inside the
    `with` block once they are known.
    """

    name: str
    attrs: dict[str, Any] = field(default_factory=dict)
    duration_s: float | None = None
    error: str | None = None

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)


class LocalTracer:
    """Collects spans in memory; flushes them to a JSON file.

    Stateful by design: one tracer per pipeline run accumulates spans,
    then `flush(path)` writes them. Safe to use without ever flushing
    (tests just read `tracer.spans`).
    """

    backend = "local"

    def __init__(self) -> None:
        self.spans: list[Span] = []

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        span = Span(name=name, attrs=dict(attrs))
        start = time.perf_counter()
        try:
            yield span
        except Exception as e:  # record the failure on the span, then re-raise
            span.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            span.duration_s = time.perf_counter() - start
            self.spans.append(span)
            self._emit(span)

    def _emit(self, span: Span) -> None:
        """Hook for backends that ship spans somewhere. Local: no-op."""

    def flush(self, path: Path) -> None:
        import json

        payload = [
            {
                "name": s.name,
                "duration_s": s.duration_s,
                "error": s.error,
                "attrs": s.attrs,
            }
            for s in self.spans
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        logger.info("wrote %d spans to %s", len(self.spans), path)


class LangfuseTracer(LocalTracer):
    """LocalTracer that also ships each span to a Langfuse server.

    Subclasses LocalTracer so the JSON baseline still gets written from
    the same spans; the only addition is `_emit`, which records the span
    as a Langfuse event. We keep the local copy unconditionally so the
    baseline does not depend on Langfuse being reachable.
    """

    backend = "langfuse"

    def __init__(self, client: Any) -> None:
        super().__init__()
        self._client = client

    def _emit(self, span: Span) -> None:
        try:
            self._client.event(
                name=span.name,
                metadata={"duration_s": span.duration_s, "error": span.error, **span.attrs},
            )
        except Exception as e:  # never let tracing break the pipeline
            logger.warning("Langfuse emit failed for span %s: %s", span.name, e)


def get_tracer() -> LocalTracer:
    """Return a Langfuse-backed tracer if configured, else a local one.

    Configured means: the `langfuse` SDK imports AND both
    LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set. Anything else
    (no SDK, no keys) falls back to LocalTracer with an INFO line, so a
    local-only run just works.
    """
    public = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret = os.getenv("LANGFUSE_SECRET_KEY")
    if not (public and secret):
        logger.info("Langfuse keys not set; using local JSON tracer.")
        return LocalTracer()
    try:
        from langfuse import Langfuse  # type: ignore
    except ImportError:
        logger.info("langfuse SDK not installed; using local JSON tracer.")
        return LocalTracer()
    logger.info("Langfuse keys found; tracing to Langfuse + local JSON.")
    return LangfuseTracer(Langfuse(public_key=public, secret_key=secret))


# Thin wrappers so callers get one-liners and the pure functions stay pure.


def trace_retrieval(tracer: LocalTracer, name: str, query: str, fn):
    """Run a retrieval callable inside a span, recording returned ids.

    `fn` is a zero-arg callable returning a list of objects with `.id`
    (RetrievedChunk). The retriever itself is unchanged; this is the
    decoration layer.
    """
    with tracer.span(name, kind="retrieval", query=query) as span:
        chunks = fn()
        span.set(returned_ids=[c.id for c in chunks], k=len(chunks))
        return chunks


async def trace_generation(tracer: LocalTracer, query: str, prompt: str, fn):
    """Await a generation callable inside a span, recording token usage.

    `fn` is a zero-arg async callable returning a `CompletionResult`.
    The provider call is unchanged; the span captures latency (from the
    `with` timing) and the token counts the provider surfaced.
    """
    with tracer.span(
        "generation", kind="generation", query=query, prompt_chars=len(prompt)
    ) as span:
        result = await fn()
        span.set(
            answer_chars=len(result.text),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_duration_s=result.total_duration_s,
        )
        return result
