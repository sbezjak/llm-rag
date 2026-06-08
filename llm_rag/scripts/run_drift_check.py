"""Re-run the baseline query set and diff it against baseline_day1.json.

This is the "day 7" half of PLAN.md's brief ("run the same 20 queries on
day 1 and day 7, then compare"). It reuses capture_report() from the
day-1 baseline script, so the identical measurement path runs both times,
then feeds both reports to observability.drift.diff_baselines.

"Day 7" is illustrative, not a 7-day timer: run this whenever something
could have moved the numbers (model update, index rebuild, dependency
bump), or now, to prove the machinery and measure the run-to-run wander
of a temp-0 pipeline (that wander is itself the finding).

Writes:
  reports/drift_report.json          the diff
  reports/baseline_<date>.json       the fresh snapshot (kept, so a later
                                     run can diff against this one too)

Requires a live Ollama and a built Chroma index (same as the baseline).
Run:

    uv run python -m llm_rag.scripts.run_drift_check
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date

from llm_rag.observability.drift import diff_baselines, format_summary
from llm_rag.scripts.run_day1_baseline import ROOT, capture_report

logger = logging.getLogger(__name__)

BASELINE_PATH = ROOT / "reports" / "baseline_day1.json"
DRIFT_REPORT_PATH = ROOT / "reports" / "drift_report.json"


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not BASELINE_PATH.exists():
        raise SystemExit(
            f"no baseline at {BASELINE_PATH}; run run_day1_baseline first to capture day 1"
        )
    old = json.loads(BASELINE_PATH.read_text())

    today = date.today().isoformat()
    snapshot_path = ROOT / "reports" / f"baseline_{today}.json"
    trace_path = ROOT / "reports" / f"baseline_{today}_trace.json"

    logger.info("capturing fresh snapshot to diff against %s", BASELINE_PATH.name)
    new = await capture_report(trace_path=trace_path)
    snapshot_path.write_text(json.dumps(new, indent=2) + "\n")
    logger.info("wrote %s", snapshot_path)

    diff = diff_baselines(old, new)
    DRIFT_REPORT_PATH.write_text(json.dumps(diff, indent=2) + "\n")
    logger.info("wrote %s", DRIFT_REPORT_PATH)

    print()
    print(format_summary(diff))
    # Non-zero exit on regression so a CI step or `&&` chain can gate on it.
    raise SystemExit(1 if diff["has_regression"] else 0)


if __name__ == "__main__":
    asyncio.run(main())
