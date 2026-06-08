"""Diff two observability baselines and flag drift.

PLAN.md's project brief says: "Run the same 20 queries on day 1 and day 7,
then compare: average latency, token cost per query, response quality
drift, any queries that suddenly started failing." S6 captured the day-1
side (reports/baseline_day1.json). This is the comparison side.

The honest subtlety: with the model, temperature (0) and inputs all held
fixed, two runs are NOT identical. Ollama generation is not bit
reproducible and the LLM judge is itself a second model call, so latency,
tokens and even judge scores wander run to run. A naive equality diff
would report "drift" every single time. So the diff is tolerance aware:
it separates expected wander (noise) from a real regression, and it
reports the raw deltas regardless so a human can see the wander itself,
that wander IS the finding (it sets the floor for what drift means here).

Pure and I/O free: it takes two report dicts (baseline_day1.json schema)
and returns a diff dict. The script layer does the file reading/writing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tolerances:
    """What counts as noise vs. a flagged regression.

    Defaults are deliberately loose on latency (it swings with machine
    load), tight on tokens (at temp 0 the prompt is fixed, so a token
    shift means the prompt builder or model changed, not noise) and
    moderate on quality. quality_floor is the absolute "this query is
    failing" bar used for the "suddenly started failing" check.
    """

    latency_pct: float = 0.50  # 50% swing in mean/p50/p95 before flagging
    tokens_pct: float = 0.10  # 10% shift in mean tokens/query
    scorer_drop: float = 0.05  # absolute drop in a scorer mean
    per_query_scorer_drop: float = 0.15  # absolute per-query score drop
    quality_floor: float = 0.50  # judge below this == "failing"


def _pct_change(old: float, new: float) -> float | None:
    """Signed fractional change, None if old is 0 (avoid div-by-zero)."""
    if old == 0:
        return None
    return round((new - old) / old, 4)


def _delta_block(old: dict, new: dict, keys: list[str], pct_tol: float) -> dict:
    """Per-key {old, new, delta, pct_change, flagged} for a metric group."""
    out = {}
    for k in keys:
        o = old.get(k)
        n = new.get(k)
        if o is None or n is None:
            out[k] = {"old": o, "new": n, "delta": None, "pct_change": None, "flagged": False}
            continue
        pct = _pct_change(o, n)
        out[k] = {
            "old": o,
            "new": n,
            "delta": round(n - o, 4),
            "pct_change": pct,
            "flagged": pct is not None and abs(pct) > pct_tol,
        }
    return out


def diff_baselines(old: dict, new: dict, tol: Tolerances | None = None) -> dict:
    """Compare two baseline reports. Returns a diff dict with metric
    deltas, per-query regressions, and a top-level `has_regression` flag
    that is True iff anything crossed a tolerance (so a test or CI step
    can gate on one boolean)."""
    tol = tol or Tolerances()

    gen = _delta_block(
        old.get("generation_latency_s", {}),
        new.get("generation_latency_s", {}),
        ["p50", "p95", "mean"],
        tol.latency_pct,
    )
    ret = _delta_block(
        old.get("retrieval_latency_s", {}),
        new.get("retrieval_latency_s", {}),
        ["p50", "p95", "mean"],
        tol.latency_pct,
    )
    tokens = _delta_block(
        old.get("tokens_per_query", {}),
        new.get("tokens_per_query", {}),
        ["mean_total", "mean_prompt", "mean_completion"],
        tol.tokens_pct,
    )

    # Scorer means: only a DROP beyond tolerance is a regression (a rise
    # is good news, not drift to flag).
    old_sc = old.get("scorer_means", {})
    new_sc = new.get("scorer_means", {})
    scorers = {}
    for k in ("semantic", "rouge", "judge"):
        o, n = old_sc.get(k), new_sc.get(k)
        if o is None or n is None:
            scorers[k] = {"old": o, "new": n, "delta": None, "flagged": False}
            continue
        delta = round(n - o, 4)
        scorers[k] = {
            "old": o,
            "new": n,
            "delta": delta,
            "flagged": delta < -tol.scorer_drop,
        }

    per_query = _diff_queries(old.get("per_query", []), new.get("per_query", []), tol)

    metric_flagged = any(v["flagged"] for v in gen.values())
    metric_flagged |= any(v["flagged"] for v in ret.values())
    metric_flagged |= any(v["flagged"] for v in tokens.values())
    metric_flagged |= any(v["flagged"] for v in scorers.values())
    query_flagged = bool(
        per_query["newly_failing"] or per_query["regressed"] or per_query["dropped_queries"]
    )

    return {
        "old_captured": old.get("captured"),
        "new_captured": new.get("captured"),
        "model": new.get("model"),
        "generation_latency_s": gen,
        "retrieval_latency_s": ret,
        "tokens_per_query": tokens,
        "scorer_means": scorers,
        "per_query": per_query,
        "has_regression": metric_flagged or query_flagged,
    }


def _diff_queries(old_rows: list[dict], new_rows: list[dict], tol: Tolerances) -> dict:
    """Per-query view: newly failing, regressed scores, and queries that
    appeared/vanished between runs."""
    old_by_id = {r["id"]: r for r in old_rows}
    new_by_id = {r["id"]: r for r in new_rows}

    newly_failing: list[dict] = []
    regressed: list[dict] = []

    for qid, new_row in new_by_id.items():
        old_row = old_by_id.get(qid)
        if old_row is None:
            continue
        old_judge = old_row.get("scores", {}).get("judge")
        new_judge = new_row.get("scores", {}).get("judge")

        # "Suddenly started failing": passed the floor before, below it now.
        if (
            old_judge is not None
            and new_judge is not None
            and old_judge >= tol.quality_floor
            and new_judge < tol.quality_floor
        ):
            newly_failing.append(
                {"id": qid, "query": new_row.get("query"), "old": old_judge, "new": new_judge}
            )

        # Any scorer dropping beyond the per-query tolerance.
        for scorer in ("semantic", "rouge", "judge"):
            o = old_row.get("scores", {}).get(scorer)
            n = new_row.get("scores", {}).get(scorer)
            if o is None or n is None:
                continue
            if (n - o) < -tol.per_query_scorer_drop:
                regressed.append(
                    {
                        "id": qid,
                        "query": new_row.get("query"),
                        "scorer": scorer,
                        "old": o,
                        "new": n,
                        "delta": round(n - o, 4),
                    }
                )

    return {
        "newly_failing": newly_failing,
        "regressed": regressed,
        "added_queries": sorted(set(new_by_id) - set(old_by_id)),
        "dropped_queries": sorted(set(old_by_id) - set(new_by_id)),
    }


def format_summary(diff: dict) -> str:
    """Human-readable one-screen summary for stdout. Mirrors the JSON but
    reads like a status line a person scans, not a blob they parse."""
    lines: list[str] = []
    verdict = "REGRESSION" if diff["has_regression"] else "no drift beyond tolerance"
    lines.append(f"Drift check: {diff['old_captured']} -> {diff['new_captured']}  [{verdict}]")
    lines.append("")

    def metric_lines(title: str, block: dict, unit: str) -> None:
        lines.append(title)
        for k, v in block.items():
            if v["old"] is None or v["new"] is None:
                lines.append(f"  {k:<14} n/a")
                continue
            pct = v.get("pct_change")
            pct_str = f"{pct * 100:+.1f}%" if pct is not None else "  n/a"
            flag = "  <-- flagged" if v["flagged"] else ""
            lines.append(f"  {k:<14} {v['old']}{unit} -> {v['new']}{unit}  ({pct_str}){flag}")

    metric_lines("generation latency", diff["generation_latency_s"], "s")
    metric_lines("retrieval latency", diff["retrieval_latency_s"], "s")
    metric_lines("tokens/query", diff["tokens_per_query"], "")

    lines.append("scorer means")
    for k, v in diff["scorer_means"].items():
        if v["old"] is None or v["new"] is None:
            lines.append(f"  {k:<14} n/a")
            continue
        flag = "  <-- dropped" if v["flagged"] else ""
        lines.append(f"  {k:<14} {v['old']} -> {v['new']}  ({v['delta']:+}){flag}")

    pq = diff["per_query"]
    lines.append("")
    lines.append(f"newly failing: {len(pq['newly_failing'])}")
    for r in pq["newly_failing"]:
        lines.append(f"  {r['id']}: judge {r['old']} -> {r['new']}  ({r['query']})")
    lines.append(f"regressed scores: {len(pq['regressed'])}")
    for r in pq["regressed"]:
        lines.append(f"  {r['id']} {r['scorer']}: {r['old']} -> {r['new']}  ({r['query']})")
    if pq["added_queries"]:
        lines.append(f"added queries: {', '.join(pq['added_queries'])}")
    if pq["dropped_queries"]:
        lines.append(f"dropped queries: {', '.join(pq['dropped_queries'])}")

    return "\n".join(lines)
