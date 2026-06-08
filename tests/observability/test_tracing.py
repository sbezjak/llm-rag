"""Mocked tests for the observability layer.

Two properties matter and both are deterministic, so no Ollama needed:

1. Tracing is real decoration: a span records latency + the attributes
   the caller set, and the wrappers capture retrieval ids / generation
   tokens. This is what feeds the day-1 baseline.
2. Tracing never breaks the pipeline: with no Langfuse keys we get the
   local tracer, and a Langfuse emit failure is swallowed.

Provider token surfacing is tested here too (respx), since the baseline
reads tokens through `complete`.
"""

from __future__ import annotations

import pytest
import respx
from httpx import Response

from llm_rag.observability import LocalTracer, get_tracer, trace_generation, trace_retrieval
from llm_rag.observability.tracing import LangfuseTracer
from llm_rag.providers.base import CompletionResult
from llm_rag.providers.ollama import OllamaProvider
from llm_rag.retrievers.vector import RetrievedChunk


def _chunk(cid: str) -> RetrievedChunk:
    return RetrievedChunk(id=cid, text="t", score=1.0, source="s", section="sec")


@pytest.mark.mocked
def test_span_records_duration_and_attrs():
    tracer = LocalTracer()
    with tracer.span("retrieval", query="q") as span:
        span.set(returned_ids=["a::b::0"])
    assert len(tracer.spans) == 1
    recorded = tracer.spans[0]
    assert recorded.name == "retrieval"
    assert recorded.attrs == {"query": "q", "returned_ids": ["a::b::0"]}
    assert recorded.duration_s is not None and recorded.duration_s >= 0.0
    assert recorded.error is None


@pytest.mark.mocked
def test_span_records_error_then_reraises():
    tracer = LocalTracer()
    with pytest.raises(ValueError):
        with tracer.span("boom"):
            raise ValueError("kaboom")
    # The failure is recorded on the span even though it re-raised.
    assert tracer.spans[0].error == "ValueError: kaboom"
    assert tracer.spans[0].duration_s is not None


@pytest.mark.mocked
def test_trace_retrieval_captures_ids():
    tracer = LocalTracer()
    chunks = trace_retrieval(
        tracer, "retrieval", "q", lambda: [_chunk("x::y::0"), _chunk("x::y::1")]
    )
    assert [c.id for c in chunks] == ["x::y::0", "x::y::1"]
    span = tracer.spans[-1]
    assert span.attrs["returned_ids"] == ["x::y::0", "x::y::1"]
    assert span.attrs["k"] == 2
    assert span.attrs["kind"] == "retrieval"


@pytest.mark.mocked
async def test_trace_generation_captures_tokens():
    tracer = LocalTracer()

    async def fake_complete() -> CompletionResult:
        return CompletionResult(
            text="hi [x::y::0]", prompt_tokens=11, completion_tokens=4, total_duration_s=0.5
        )

    result = await trace_generation(tracer, "q", "the prompt", fake_complete)
    assert result.text == "hi [x::y::0]"
    span = tracer.spans[-1]
    assert span.attrs["prompt_tokens"] == 11
    assert span.attrs["completion_tokens"] == 4
    assert span.attrs["prompt_chars"] == len("the prompt")


@pytest.mark.mocked
def test_flush_writes_json(tmp_path):
    tracer = LocalTracer()
    with tracer.span("retrieval", query="q") as s:
        s.set(returned_ids=["a"])
    out = tmp_path / "trace.json"
    tracer.flush(out)
    import json

    payload = json.loads(out.read_text())
    assert payload[0]["name"] == "retrieval"
    assert payload[0]["attrs"]["returned_ids"] == ["a"]


@pytest.mark.mocked
def test_get_tracer_falls_back_to_local_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    tracer = get_tracer()
    assert isinstance(tracer, LocalTracer)
    assert tracer.backend == "local"


@pytest.mark.mocked
def test_langfuse_emit_failure_does_not_break_span():
    class BrokenClient:
        def event(self, **kwargs):
            raise RuntimeError("network down")

    tracer = LangfuseTracer(BrokenClient())
    # Span completes and is recorded despite the emit raising internally.
    with tracer.span("retrieval", query="q"):
        pass
    assert len(tracer.spans) == 1
    assert tracer.spans[0].error is None  # span itself did not fail


@pytest.mark.mocked
@respx.mock
async def test_provider_surfaces_token_counts():
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=Response(
            200,
            json={
                "response": "hello",
                "prompt_eval_count": 42,
                "eval_count": 7,
                "total_duration": 1_500_000_000,  # 1.5s in ns
            },
        )
    )
    result = await OllamaProvider().complete("hi")
    assert result.text == "hello"
    assert result.prompt_tokens == 42
    assert result.completion_tokens == 7
    assert result.total_duration_s == pytest.approx(1.5)


@pytest.mark.mocked
@respx.mock
async def test_provider_complete_tolerates_missing_token_fields():
    respx.post("http://localhost:11434/api/generate").mock(
        return_value=Response(200, json={"response": "hello"})
    )
    result = await OllamaProvider().complete("hi")
    assert result.text == "hello"
    assert result.prompt_tokens is None
    assert result.completion_tokens is None
    assert result.total_duration_s is None
