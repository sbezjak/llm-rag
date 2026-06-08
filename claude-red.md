# CLAUDE.md

> STARTER for project 3 (red team). Copy into the new repo and rename to
> `CLAUDE.md`. Sections marked TODO are red-team-specific - fill them in.
> Everything under "Working style with this user" is reusable verbatim.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Pytest-based **red team test suite** for an LLM-backed API. Targets the
existing **FastAPI + LLM project** (project 0,
https://github.com/sbezjak/llm-api-testing) as the system under test, not
a fresh app - the point is to try to break something that already exists.
Python 3.11+, managed with `uv`.

**One focused theme:** TODO - state the single failure mode this project is
about, e.g. "which adversarial inputs get through an LLM API's defenses,
and how do you tell a real bypass from the model just being chatty?" See
`PLAN.md` for full scope discipline.

Package layout under `llm_redteam/` (TODO: confirm package name, pin in
`pyproject.toml` `packages = [...]` for the wheel build):

- `providers/`, adapters that talk to the target API / LLM backend. The
  only place that issues HTTP calls; tests mock at this boundary with
  `respx`. Shape ported from the eval harness / RAG projects.
- `attacks/`, TODO - the adversarial payload catalog, grouped by attack
  family (direct injection, indirect injection, system-prompt extraction,
  jailbreak/role-play, encoded/obfuscated, multilingual). Pure data +
  light construction, no network calls.
- `probes/` or `runners/`, TODO - send an attack at the target and capture
  the raw response. Deterministic given (attack, target response).
- `detectors/`, TODO - decide whether a response indicates a *bypass*
  (leaked system prompt, followed the injected instruction, produced
  disallowed content). This is the red-team analog of the RAG scorers and
  is where most of the judgment lives - keep it I/O-free except for any
  LLM-judge detector, which calls a provider.
- `severity.py`, TODO - map a detected bypass to a severity tier.
- `dataset.py`, loads + validates the attack datasets under `data/`.

Datasets live at `data/` (TODO):

- `attacks/`, the payloads, one file per attack family.
- `expected.yaml`, per attack: what a *safe* response looks like vs a
  *bypass*, so pass/fail is checkable.

## Commands

Use `uv` for all environment and execution tasks:

- Install / sync deps (including dev group): `uv sync`
- Run all tests: `uv run pytest`
- Run a single test: `uv run pytest tests/path/to/test_file.py::test_name`
- Run only fast (mocked) tests: `uv run pytest -m mocked`
- Run only tests that hit the live target API: `uv run pytest -m live` (TODO: confirm marker name)
- Skip live tests: `uv run pytest -m "not live"`
- HTML report: auto-generated at `reports/report.html` on every `uv run pytest` (self-contained, includes captured logs at INFO+).
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`

Project-specific scripts (add incrementally across sessions): TODO.

## Test conventions (configured in pyproject.toml)

- `asyncio_mode = "auto"`, async tests do not need `@pytest.mark.asyncio`.
- Custom markers gate environment-dependent tests:
  - `@pytest.mark.live`, slow, requires the live target API running. (TODO: confirm name)
  - `@pytest.mark.mocked`, uses `respx` to mock the target HTTP API;
    should be the default for unit tests.
- `testpaths = ["tests"]`; `ruff` line length is 100, target `py311`.

Test categories (mirror in `tests/` subfolders) - TODO, adapt to attack
families. Keeping families separate is how a single broad bypass doesn't
get mistaken for a category-wide failure:

- `tests/injection_direct/`, "ignore previous instructions" style.
- `tests/injection_indirect/`, hostile instructions hidden in content the
  system ingests.
- `tests/prompt_extraction/`, attempts to leak the system prompt.
- `tests/jailbreak/`, role-play / persona attacks.
- `tests/obfuscation/`, encoded / Unicode-trick / multilingual attacks.

Use the `xfail(strict=True)`-as-contract pattern (ported from projects 1-2)
to lock in *known* bypasses: a strict xfail that fails loudly if a bypass
ever stops working, forcing acknowledgment instead of silent drift.

## Architecture intent

Preserve these seams when adding code:

- HTTP calls only inside `providers/`. Tests mock here with `respx`.
- Attack construction has no network calls - deterministic given the
  payload data.
- Detection depends on the target's response but tests inject known
  responses as fixtures, so attack-construction and detection failures
  don't contaminate each other.
- Detectors stay I/O-free (an LLM-judge detector is the exception, it
  calls a provider).

## Working style with this user

- **Prepare drafts/templates for any task the user has to do by hand.**
  When the next step is something only the user can do (write an attack
  payload, decide whether a given response counts as a bypass, grade a
  refusal), prepare a fill-in-the-blanks file with the structure pre-built.
  Don't make the user start from a blank page. Examples: a YAML scaffold
  with TODOs, a markdown table with rows pre-filled, a notes template with
  section headings. Reduce the user's task to filling in the squishy parts.
- **Capture explanations to `notes.md` when teaching.** When the user asks
  "explain this to me" and the answer is non-trivial (why an attack family
  works, how a detector decides, OWASP LLM Top 10 mechanics), mirror it
  (lightly cleaned up) into `notes.md` as reference material. The chat
  scrolls; the article stays.
- **Two writing registers, no duplicate copies.** `notes.md` is the dense,
  finding-first record. The teaching / front-door artifacts (the narrated
  walkthrough, `docs/` explainers, README) use the plain, layered voice the
  user likes (plain words, an analogy or two, structured so a reader can
  stop as soon as they have it). Same explanation in two registers drifts
  out of sync, so the registers serve different artifacts, never two copies
  of one thing. Do not add a standalone concepts/glossary file; notes.md
  plus the walkthrough cover both lookup and teaching.
- **Default to production / best-practice solutions; take the pragmatic
  shortcut only when the trade-off is justified for this project's scope,
  and call the trade-off out explicitly.** Don't silently pick the easy
  path. Name what the production-grade pattern would be, name why we're not
  doing it here (scope, runtime context, learning focus), and write the
  trade-off into `notes.md` so the writeup shows it was a deliberate choice,
  not an oversight.
- **Validate a new integration cheaply before any long or expensive run.**
  Before launching a long run (especially one exercising a new external
  tool that has never run end to end here), prove it works on the smallest
  possible input first (1-2 items, fabricated inputs are fine) and confirm
  the output parses / scores. ALWAYS arm a monitor on the log for any run
  that is not near-instant - grepping for success AND failure signatures
  (`500|Internal Server Error|Traceback|Error|Exception|Killed|OOM|NaN|Timeout|Failed`)
  so a mid-run failure surfaces at the failing job. Smoke test first,
  monitor second, both every time. When a run does fail, fix the root cause
  and re-validate cheaply before re-running full.
- **Always make model prompts and responses visible in test reports.**
  Every component that calls a model (providers, LLM-judge detectors) must
  log the prompt going in and the response coming out at `INFO` level via
  the stdlib `logging` module, not truncated. The always-on pytest-html
  report captures these, so any failing test's "Captured log" section shows
  exactly what the model saw and said. For a red-team project this is how
  you tell a real bypass from the model just being verbose.
- **Prefer the smallest solution that solves the problem; after writing,
  re-read and cut machinery the task didn't ask for.** Asked to "add
  validation", don't reach for a configurable framework with custom
  exception classes when a few lines of `if` are the real answer. Treat the
  re-read as a step: "is half of this scaffolding I invented but nobody
  asked for?" If yes, rewrite it small.
- In all prose (docs, comments, commit messages, PR descriptions), join
  clauses with a single hyphen `-`, a comma, a period, or parentheses. The
  only dash character in written text is a single `-`.
- End commit messages at the body. The user is the sole author.
