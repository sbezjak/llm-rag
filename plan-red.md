# PLAN.md

> STARTER for project 3 (red team). Copy into the new repo and rename to
> `PLAN.md`. Part 1, the scope-discipline lessons, the capstone note, and
> the Article recipe are REUSABLE - update names, keep the shape. Part 2 is
> a template with TODOs for the red-team-specific scope.

> **In plain English:** this file is the planning notebook for the project.
> It says what the project is, what I deliberately decided *not* to do (and
> why), how the work is broken into sessions, and what lessons I carry into
> the next project. The **README** is the friendlier door in; PLAN.md is
> the back office.

---

## START HERE (do this before writing any code)

1. **Run the scaffolding skill first.** Invoke `scaffold-llm-pytest-project`
   to lay down the repo shape (markers `mocked` vs live, async-by-default
   tests, ruff config, the `Provider` over httpx, pytest-html reporting).
   Do NOT hand-build the skeleton or copy files from a previous repo - the
   discipline is "match their shape," and the skill is the shape. This is
   the step that's easy to forget.
2. Then rename `claude-red.md` -> `CLAUDE.md` and this file -> `PLAN.md`.
3. **Petri check.** Look at https://alignment.anthropic.com/2025/petri/ and
   decide if it's usable for adversarial testing here, then tell the user.
   (Carried forward from the 5-project roadmap.)
4. Fill in the TODOs in Part 2 below, then lock scope before session 1.

---

# Part 1, Portfolio context (REUSABLE: copy into next project's PLAN.md and update)

A 5-project AI/QA learning portfolio. Each project gets its own repo, its
own writeup, and exactly one focused theme. The order matters because
**scope decisions for any one project depend on what's coming later**,
don't cram theme-N material into project-M.

## Past projects (with links)

| # | Project | Theme | Repo | Status |
|---|---|---|---|---|
| 0 | LLM API Testing | How to test a non-deterministic FastAPI + LLM API | https://github.com/sbezjak/llm-api-testing | shipped, quality bar |
| 1 | LLM Eval Harness | How to score non-deterministic LLM output (5 scorers, calibration, bias) | https://github.com/sbezjak/llm-eval-harness | shipped |
| 2 | RAG + Observability | Retrieval correctness, hallucination, monitoring, drift | TODO: add repo url | shipped |
| 3 | Red Team Test Suite | Prompt injection, jailbreaks, OWASP LLM Top 10 | *this repo* | in progress |
| 4 | Agent Testing | Tool use, decision chains, multi-step traces | TBD | future |
| 5 | Model Benchmarking | Cost/latency/quality across providers (uses project 1's harness) | TBD | future |
| M | Project Master (capstone) | Connect P1-P5 via a shared eval + observability spine; one comparable scoreboard | TBD | deferred |

**Project 0 is the quality bar for every later project.** Same README
shape, same hosted-reports discipline, same marker conventions (`mocked` vs
live), same writeup style. Don't import from previous projects, match their
shape.

## Scope-discipline lessons (apply to every project, these are non-negotiable)

Distilled from earlier projects, where scope crept further than intended.
Each one is a real failure mode that already happened, not a hypothetical.

1. **Quality > size, always.** 10 deliberately-adversarial test items beat
   50 mostly-easy ones for a *learning* writeup. Set a stopping rule by
   **number of distinct findings**, not number of items or tests. If 10
   items produce 9 findings, ship.
2. **Decide a stopping rule up front, not after the fact.** Before starting
   a session arc, write: "I ship when I have N findings" or "I ship after
   session S, regardless." In-the-moment momentum from one finding makes the
   next feel cheap (project 1 grew 3 -> 6.5 sessions this way).
3. **Every addition needs a justified trade-off, written down.** Take the
   production-grade pattern by default. If you take the shortcut, name what
   production would do, name why not here, write it into `notes.md` / `docs/`.
4. **One theme per project.** If a finding from one project pre-justifies a
   decision in another, *write a forward pointer*, don't build the thing.
5. **When work has a natural future follow-up, defer it explicitly.**
   "Deferred to project N because lesson L lands in that context" beats
   "we'll figure that out later."
6. **Public-facing files stand alone.** No "Finding N" cross-references, no
   internal jargon, no references to private notes. README + docstrings +
   test names read plainly to a QA engineer who has never seen the repo.
7. **One repo per project, one writeup per project.** No CI across 5 repos.
   Hosted reports + a clean README is the deliverable.

## Project M, capstone (deferred, do NOT build during P1-P5)

After all five ship, a 6th "capstone" repo connects them as the integration
layer. Recorded so the idea isn't lost; building it now violates discipline
lesson 4. What the five already share: a `Provider` over httpx + Ollama, the
`Scorer.score(...) -> ScoreResult` contract, the `mocked` / live marker
discipline, pytest-html reporting, and the fact that every project produces
traces (prompt in, response out, scored). **Red-team relevance:** P3's
attack suite is a planned cross-project flow in the capstone (run the same
attacks against the RAG and agent projects), so build the attack catalog so
it can be pointed at any target, not just project 0.

---

# Part 2, This project (project 3: Red Team Test Suite) — TEMPLATE, fill in

## What this project is

A pytest-based adversarial test suite run against the **existing FastAPI +
LLM API (project 0)**. Not a new app - the point is to break something that
already exists.

**One focused theme:** TODO - the single failure mode, e.g. "which
adversarial inputs slip past an LLM API's guardrails, and how do you tell a
genuine bypass from the model just being chatty?"

## Stopping rule (decide BEFORE starting)

TODO - pick one and write it here now: "ship when I have N distinct
findings" or "ship after session S regardless." (See discipline lesson 2.)

## Scope (lock at start)

TODO - the artifacts that will exist when this ships. Likely:

- **Attack catalog:** payloads grouped by family (direct injection,
  indirect injection, system-prompt extraction, jailbreak/role-play,
  encoded/obfuscated, multilingual).
- **Detectors:** decide bypass vs safe per response; one LLM-judge detector
  for the squishy cases.
- **Severity tiers:** map each bypass to a severity.
- **Test suite:** categories mirrored from the attack families.
- **Writeup artifacts:** private `notes.md` (findings), public `docs/`
  walkthrough + article, README at project-0 bar, hosted reports.

## Pre-staked decisions (don't relitigate)

1. **Project 3 extends project 0 (FastAPI + LLM API), not the RAG repo.**
   Don't bring RAG into red-team. If a red-team finding implies a RAG
   hardening, write a forward pointer.
2. **Petri check is a START HERE step**, not a mid-project discovery.
3. **Reuse the `xfail(strict=True)`-as-contract pattern** to lock in known
   bypasses. The mechanism transferred across projects 1-2; don't reinvent.
4. TODO - add red-team-specific pre-stakes as you make them (which attack
   families are in scope vs deferred, severity model, etc.).

---

# Article recipe (REUSABLE: the public writeup shape for every project)

Each project ships two writeups, in two registers (see CLAUDE.md
"two writing registers"):

- `docs/walkthrough.md`, findings-first, complete, the in-repo record. Read
  it to reload the project; it is the *source* the article is cut from.
- `docs/<project>-article.md`, the public narrative, derived from the
  walkthrough. The thing posted (LinkedIn + dev site). Naming is
  `eval-article.md`, `rag-article.md`, `redteam-article.md`, etc.

**Drafting workflow (the rhythm that worked on project 2):** go section by
section - edit and talk through one section, settle it, then move to the
next. Finish `walkthrough.md` completely before opening the article, so the
article is cut from settled material instead of drifting alongside it.

**Keep this signature constant across all five articles** (it's what makes
the series read like one author with one method):

- Plain first-person voice, concrete numbers inline, single hyphens only.
- Opening that frames *this* project against the series ("Project N was X,
  this one is Y") and states the one failure mode it's about.
- The line: "A learning project, written up for anyone trying to get into
  AI testing."
- `Repo: <url>` near the top and again at the bottom.
- The honesty move: lead with the single strongest finding told as a story;
  show where each thing was *wrong*, not where it worked.
- Closing "Project N of five..." pointer that links the prior project and
  names the next.

**Keep this arc constant:** strongest finding as a story -> supporting
findings -> "smaller findings, briefly" -> "the thing I'd take back into a
normal test suite tomorrow" (one reusable technique) -> "a note on the
numbers" (honest limitations) -> "how to run it" -> conclusion.

**Flex the body to the project's actual story, don't force an identical
section list.** The signature and arc are fixed; group the findings to match
what the project *is*. (Project 2 grouped under "when the search is wrong" /
"when the model is wrong" because it was a two-subsystems story; a red-team
project might group by attack family or by severity.)
