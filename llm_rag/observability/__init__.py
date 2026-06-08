"""Observability: tracing decoration over retrievers and generators.

Decoration only, never wired inside the pure layers. See tracing.py.
"""

from .tracing import (
    LangfuseTracer,
    LocalTracer,
    Span,
    get_tracer,
    trace_generation,
    trace_retrieval,
)

__all__ = [
    "LangfuseTracer",
    "LocalTracer",
    "Span",
    "get_tracer",
    "trace_generation",
    "trace_retrieval",
]
