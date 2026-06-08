"""Drift diff is pure, so it tests without Ollama or Chroma: feed two
synthetic baselines and assert the deltas and the regression flags.

This protects the comparison logic itself, the part that decides whether
"day 7" looks different from "day 1". The live capture is exercised by
the baseline/observability run; here we only care that given two reports,
the diff math and the tolerance gates are right.
"""

from __future__ import annotations

import pytest

from llm_rag.observability.drift import Tolerances, diff_baselines, format_summary

pytestmark = pytest.mark.mocked


def _baseline(**overrides) -> dict:
    """A minimal day-1-shaped report; overrides patch nested keys."""
    base = {
        "captured": "2026-05-27",
        "model": "llama3.2",
        "generation_latency_s": {"p50": 4.0, "p95": 10.0, "mean": 5.0},
        "retrieval_latency_s": {"p50": 1.6, "p95": 2.6, "mean": 1.8},
        "tokens_per_query": {"mean_total": 460.0, "mean_prompt": 386.0, "mean_completion": 74.0},
        "scorer_means": {"semantic": 0.83, "rouge": 0.32, "judge": 0.87},
        "per_query": [
            {
                "id": "gen-001",
                "query": "q1",
                "scores": {"semantic": 0.84, "rouge": 0.33, "judge": 0.85},
            },
            {
                "id": "gen-002",
                "query": "q2",
                "scores": {"semantic": 0.88, "rouge": 0.30, "judge": 0.85},
            },
        ],
    }
    for k, v in overrides.items():
        base[k] = v
    return base


def test_identical_reports_show_no_regression():
    diff = diff_baselines(_baseline(), _baseline())
    assert diff["has_regression"] is False
    assert diff["generation_latency_s"]["mean"]["delta"] == 0.0
    assert diff["per_query"]["newly_failing"] == []


def test_small_wander_within_tolerance_is_not_flagged():
    # +20% latency, +3% tokens, judge -0.02: all under default tolerances.
    new = _baseline(
        generation_latency_s={"p50": 4.8, "p95": 11.0, "mean": 6.0},
        tokens_per_query={"mean_total": 474.0, "mean_prompt": 386.0, "mean_completion": 74.0},
        scorer_means={"semantic": 0.83, "rouge": 0.32, "judge": 0.85},
    )
    diff = diff_baselines(_baseline(), new)
    assert diff["has_regression"] is False
    assert diff["generation_latency_s"]["mean"]["flagged"] is False


def test_latency_blowup_is_flagged():
    new = _baseline(generation_latency_s={"p50": 4.0, "p95": 10.0, "mean": 9.0})  # +80%
    diff = diff_baselines(_baseline(), new)
    assert diff["generation_latency_s"]["mean"]["flagged"] is True
    assert diff["has_regression"] is True


def test_token_shift_is_flagged_tightly():
    # +15% tokens: under latency's 50% bar, over tokens' 10% bar.
    new = _baseline(
        tokens_per_query={"mean_total": 529.0, "mean_prompt": 386.0, "mean_completion": 74.0}
    )
    diff = diff_baselines(_baseline(), new)
    assert diff["tokens_per_query"]["mean_total"]["flagged"] is True
    assert diff["has_regression"] is True


def test_scorer_rise_is_not_a_regression():
    new = _baseline(scorer_means={"semantic": 0.90, "rouge": 0.40, "judge": 0.95})
    diff = diff_baselines(_baseline(), new)
    assert diff["scorer_means"]["judge"]["flagged"] is False
    assert diff["has_regression"] is False


def test_scorer_mean_drop_is_flagged():
    new = _baseline(scorer_means={"semantic": 0.70, "rouge": 0.32, "judge": 0.87})  # -0.13
    diff = diff_baselines(_baseline(), new)
    assert diff["scorer_means"]["semantic"]["flagged"] is True
    assert diff["has_regression"] is True


def test_query_that_started_failing_is_caught():
    new = _baseline(
        per_query=[
            {
                "id": "gen-001",
                "query": "q1",
                "scores": {"semantic": 0.84, "rouge": 0.33, "judge": 0.40},
            },
            {
                "id": "gen-002",
                "query": "q2",
                "scores": {"semantic": 0.88, "rouge": 0.30, "judge": 0.85},
            },
        ]
    )
    diff = diff_baselines(_baseline(), new)
    failing = diff["per_query"]["newly_failing"]
    assert [r["id"] for r in failing] == ["gen-001"]
    assert diff["has_regression"] is True


def test_per_query_score_drop_is_flagged_even_above_floor():
    # judge 0.85 -> 0.60: still above the 0.50 floor, but a 0.25 drop.
    new = _baseline(
        per_query=[
            {
                "id": "gen-001",
                "query": "q1",
                "scores": {"semantic": 0.84, "rouge": 0.33, "judge": 0.60},
            },
            {
                "id": "gen-002",
                "query": "q2",
                "scores": {"semantic": 0.88, "rouge": 0.30, "judge": 0.85},
            },
        ]
    )
    diff = diff_baselines(_baseline(), new)
    regressed = diff["per_query"]["regressed"]
    assert any(r["id"] == "gen-001" and r["scorer"] == "judge" for r in regressed)
    assert "gen-001" not in [r["id"] for r in diff["per_query"]["newly_failing"]]


def test_added_and_dropped_queries_are_reported():
    new = _baseline(
        per_query=[
            {
                "id": "gen-002",
                "query": "q2",
                "scores": {"semantic": 0.88, "rouge": 0.30, "judge": 0.85},
            },
            {
                "id": "gen-003",
                "query": "q3",
                "scores": {"semantic": 0.80, "rouge": 0.30, "judge": 0.85},
            },
        ]
    )
    diff = diff_baselines(_baseline(), new)
    assert diff["per_query"]["added_queries"] == ["gen-003"]
    assert diff["per_query"]["dropped_queries"] == ["gen-001"]
    # A dropped query counts as drift (the set changed).
    assert diff["has_regression"] is True


def test_missing_metric_is_handled_not_crashed():
    new = _baseline(
        tokens_per_query={"mean_total": None, "mean_prompt": None, "mean_completion": None}
    )
    diff = diff_baselines(_baseline(), new)
    assert diff["tokens_per_query"]["mean_total"]["flagged"] is False


def test_custom_tolerances_tighten_gates():
    new = _baseline(generation_latency_s={"p50": 4.0, "p95": 10.0, "mean": 6.0})  # +20%
    loose = diff_baselines(_baseline(), new)
    strict = diff_baselines(_baseline(), new, Tolerances(latency_pct=0.10))
    assert loose["generation_latency_s"]["mean"]["flagged"] is False
    assert strict["generation_latency_s"]["mean"]["flagged"] is True


def test_format_summary_is_a_nonempty_string():
    diff = diff_baselines(_baseline(), _baseline())
    out = format_summary(diff)
    assert "Drift check" in out
    assert "no drift beyond tolerance" in out
